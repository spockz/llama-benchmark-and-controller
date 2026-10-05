package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRouterSwitchCacheRoundTrip(t *testing.T) {
	tmp := t.TempDir()
	presetPath := filepath.Join(tmp, "models.ini")
	writeTestPresets(t, presetPath, 4096)
	router := newCacheTestRouter()
	client := inMemoryClient(router)

	stateDir := filepath.Join(tmp, "state")
	runSwitch := testSwitcher(client, presetPath, stateDir)
	if err := runSwitch("model-b"); err != nil {
		t.Fatalf("switch A to B: %v", err)
	}
	// Keep the manifest but remove the slot file to exercise stale-file
	// fallback. A missing restore must be reported as a cold load, not failure.
	router.clearCacheFiles()
	if err := runSwitch("model-a"); err != nil {
		t.Fatalf("switch B to A: %v", err)
	}
	if err := runSwitch("model-b"); err != nil {
		t.Fatalf("switch A to B after saved B cache: %v", err)
	}
	router.assertCounts(t, "model-b", 2, 1, 0, 1)

	// A server restart loses the active model but not slot files on disk. Rebuild
	// that state here and ensure a fresh CLI invocation can restore the cache.
	router.setActive("")
	if err := runSwitch("model-a"); err != nil {
		t.Fatalf("restore model A after simulated server restart: %v", err)
	}
	if err := runSwitch("model-b"); err != nil {
		t.Fatalf("switch after simulated server restart: %v", err)
	}
	router.assertRestores(t, 1, 2)

	// Loading failure after unloading is surfaced with the actual server state
	// left discoverable instead of claiming the target is active.
	if err := runSwitch("model-a"); err != nil {
		t.Fatalf("load A before failure test: %v", err)
	}
	router.setFailLoad("model-b")
	if err := runSwitch("model-b"); err == nil || !strings.Contains(err.Error(), "load \"model-b\" after unload") {
		t.Fatalf("expected target load failure, got %v", err)
	}
	router.assertActive(t, "")
}

func TestUnloadCommandUnloadsEveryActiveModel(t *testing.T) {
	router := &unloadTestRouter{active: map[string]bool{"model-a": true, "model-b": true}}
	api, err := newAPIClient(options{serverURL: "http://router.test"}, inMemoryClient(router))
	if err != nil {
		t.Fatal(err)
	}
	opt := options{idleTimeout: time.Second, waitTimeout: time.Second, poll: time.Millisecond}
	var output strings.Builder
	err = unloadCommand(context.Background(), api, opt, map[string]preset{}, []string{"--discard-cache"}, &output)
	if err != nil {
		t.Fatalf("unload all active models: %v", err)
	}
	if len(router.unloaded) != 2 || router.active["model-a"] || router.active["model-b"] {
		t.Fatalf("unloads = %v, active models = %v", router.unloaded, router.active)
	}
	if !strings.Contains(output.String(), "All router models are unloaded; GPU is available") {
		t.Fatalf("unload confirmation missing: %s", output.String())
	}
}

type unloadTestRouter struct {
	active   map[string]bool
	unloaded []string
}

func (router *unloadTestRouter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/models":
		models := []modelRecord{
			{ID: "model-a", Status: modelStatus{Value: unloadTestState(router.active["model-a"])}},
			{ID: "model-b", Status: modelStatus{Value: unloadTestState(router.active["model-b"])}},
		}
		_ = json.NewEncoder(w).Encode(modelList{Data: models})
	case r.Method == http.MethodGet && r.URL.Path == "/slots":
		_ = json.NewEncoder(w).Encode([]slot{{ID: 0}})
	case r.Method == http.MethodPost && r.URL.Path == "/models/unload":
		var request map[string]string
		_ = json.NewDecoder(r.Body).Decode(&request)
		modelID := request["model"]
		router.active[modelID] = false
		router.unloaded = append(router.unloaded, modelID)
		_ = json.NewEncoder(w).Encode(map[string]bool{"success": true})
	default:
		http.NotFound(w, r)
	}
}

func unloadTestState(active bool) string {
	if active {
		return "loaded"
	}
	return "unloaded"
}

type cacheTestRouter struct {
	mu         sync.Mutex
	active     string
	cacheFiles map[string]bool
	restores   map[string]int
	saves      map[string]int
	busy       bool
	failLoad   string
}

func newCacheTestRouter() *cacheTestRouter {
	return &cacheTestRouter{
		active:     "model-a",
		cacheFiles: map[string]bool{},
		restores:   map[string]int{},
		saves:      map[string]int{},
	}
}

func (router *cacheTestRouter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	router.mu.Lock()
	defer router.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/models":
		router.serveModels(w)
	case r.Method == http.MethodGet && r.URL.Path == "/props":
		_ = json.NewEncoder(w).Encode(map[string]any{"model_path": r.URL.Query().Get("model") + ".gguf", "build_info": "b-test-commit"})
	case r.Method == http.MethodGet && r.URL.Path == "/slots":
		_ = json.NewEncoder(w).Encode([]slot{{ID: 0, IsProcessing: router.busy}})
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/slots/0"):
		router.serveSlot(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/models/unload":
		router.active = ""
		_ = json.NewEncoder(w).Encode(map[string]any{"success": true})
	case r.Method == http.MethodPost && r.URL.Path == "/models/load":
		router.serveLoad(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (router *cacheTestRouter) serveModels(w http.ResponseWriter) {
	data := []modelRecord{
		{ID: "model-a", Status: modelStatus{Value: "unloaded"}},
		{ID: "model-b", Status: modelStatus{Value: "unloaded"}},
	}
	for i := range data {
		if data[i].ID == router.active {
			data[i].Status.Value = "loaded"
		}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
}

func (router *cacheTestRouter) serveSlot(w http.ResponseWriter, r *http.Request) {
	var body map[string]string
	_ = json.NewDecoder(r.Body).Decode(&body)
	modelID, filename := body["model"], body["filename"]
	if r.URL.Query().Get("action") == "save" {
		router.cacheFiles[filename] = true
		router.saves[modelID]++
		_ = json.NewEncoder(w).Encode(map[string]any{"n_saved": 42, "filename": filename})
		return
	}
	if !router.cacheFiles[filename] {
		http.Error(w, "cache missing", http.StatusNotFound)
		return
	}
	router.restores[modelID]++
	_ = json.NewEncoder(w).Encode(map[string]any{"n_restored": 42})
}

func (router *cacheTestRouter) serveLoad(w http.ResponseWriter, r *http.Request) {
	var body map[string]string
	_ = json.NewDecoder(r.Body).Decode(&body)
	if body["model"] == router.failLoad {
		http.Error(w, "configured test load failure", http.StatusInternalServerError)
		return
	}
	router.active = body["model"]
	_ = json.NewEncoder(w).Encode(map[string]any{"success": true})
}

func (router *cacheTestRouter) clearCacheFiles() {
	router.mu.Lock()
	defer router.mu.Unlock()
	clear(router.cacheFiles)
}

func (router *cacheTestRouter) setActive(modelID string) {
	router.mu.Lock()
	defer router.mu.Unlock()
	router.active = modelID
}

func (router *cacheTestRouter) setFailLoad(modelID string) {
	router.mu.Lock()
	defer router.mu.Unlock()
	router.failLoad = modelID
}

func (router *cacheTestRouter) assertActive(t *testing.T, expected string) {
	t.Helper()
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.active != expected {
		t.Fatalf("active model = %q, want %q", router.active, expected)
	}
}

func (router *cacheTestRouter) assertCounts(t *testing.T, active string, savesA, savesB, restoresA, restoresB int) {
	t.Helper()
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.active != active || router.saves["model-a"] != savesA || router.saves["model-b"] != savesB || router.restores["model-a"] != restoresA || router.restores["model-b"] != restoresB {
		t.Fatalf("router state active=%q saves=%v restores=%v", router.active, router.saves, router.restores)
	}
}

func (router *cacheTestRouter) assertRestores(t *testing.T, modelA, modelB int) {
	t.Helper()
	router.mu.Lock()
	defer router.mu.Unlock()
	if router.restores["model-a"] != modelA || router.restores["model-b"] != modelB {
		t.Fatalf("restores=%v, want model-a=%d model-b=%d", router.restores, modelA, modelB)
	}
}

func testSwitcher(client *http.Client, presetPath, stateDir string) func(string) error {
	return func(target string) error {
		args := []string{"--server", "http://router.test", "--preset", presetPath, "--state-dir", stateDir, "--session", "test"}
		args = append(args, "--model-sha256", "model-a="+strings.Repeat("a", 64))
		args = append(args, "--model-sha256", "model-b="+strings.Repeat("b", 64), "switch", target)
		return runWithHTTP(args, &strings.Builder{}, &strings.Builder{}, client)
	}
}

func TestSwitchRefusesBusySlotAndLeavesServerLoaded(t *testing.T) {
	tmp := t.TempDir()
	presetPath := filepath.Join(tmp, "models.ini")
	writeTestPresets(t, presetPath, 4096)
	var mu sync.Mutex
	active := "model-a"
	busy := true
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/models":
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []modelRecord{{ID: "model-a", Status: modelStatus{Value: map[bool]string{true: "loaded", false: "unloaded"}[active == "model-a"]}}, {ID: "model-b", Status: modelStatus{Value: "unloaded"}}}})
		case "/slots":
			_ = json.NewEncoder(w).Encode([]slot{{ID: 0, IsProcessing: busy}})
		default:
			http.NotFound(w, r)
		}
	})
	err := runWithHTTP([]string{"--server", "http://router.test", "--preset", presetPath, "--state-dir", filepath.Join(tmp, "state"), "--idle-timeout", "5ms", "switch", "model-b"}, &strings.Builder{}, &strings.Builder{}, inMemoryClient(handler))
	if err == nil || !strings.Contains(err.Error(), "refusing to interrupt active generation") {
		t.Fatalf("expected refusal for busy generation, got %v", err)
	}
	if active != "model-a" {
		t.Fatalf("busy active model was changed: %s", active)
	}
}

type transportFunc func(*http.Request) (*http.Response, error)

func (fn transportFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func inMemoryClient(handler http.Handler) *http.Client {
	return &http.Client{Transport: transportFunc(func(request *http.Request) (*http.Response, error) {
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		return recorder.Result(), nil
	})}
}

func TestPresetGlobalValuesAndCompatibility(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models.ini")
	if err := os.WriteFile(path, []byte("[*]\ntemp = 0.6\n[q3]\nhf = repo:q3\nctx-size = 131072\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	presets, err := readPresets(path)
	if err != nil {
		t.Fatal(err)
	}
	if presets["q3"].Values["temp"] != "0.6" || presets["q3"].Values["ctx-size"] != "131072" {
		t.Fatalf("global and named settings were not merged: %#v", presets["q3"].Values)
	}
	one := cacheManifest{Version: 1, ModelID: "q3", Session: "s", ModelSHA256: strings.Repeat("a", 64), BuildInfo: "build", Restorable: true}
	two := one
	if !compatible(one, two) {
		t.Fatal("identical manifests should be compatible")
	}
	two.BuildInfo = "different-build"
	if compatible(one, two) {
		t.Fatal("different build identity must reject cache restoration")
	}
}

func TestExamplePresetsConfigureBothModelsAt128K(t *testing.T) {
	presets, err := readPresets(filepath.Join("..", "..", "models.ini.example"))
	if err != nil {
		t.Fatal(err)
	}
	q3 := presets["qwen36-q3-k-m"].Values
	if q3["ctx-size"] != "131072" || q3["batch-size"] != "2048" || q3["ubatch-size"] != "256" || q3["fit-target"] != "512" {
		t.Fatalf("unexpected Q3 preset: %#v", q3)
	}
	if presets["qwen36-iq4-xs"].Values["ctx-size"] != "131072" {
		t.Fatalf("IQ4_XS context is not 128K: %#v", presets["qwen36-iq4-xs"].Values)
	}
}

func writeTestPresets(t *testing.T, path string, contextSize int) {
	t.Helper()
	content := fmt.Sprintf("[model-a]\nhf = repo:a\nctx-size = %d\nslot-save-path = /tmp/\n[model-b]\nhf = repo:b\nctx-size = %d\nslot-save-path = /tmp/\n", contextSize, contextSize)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
