package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
)

func runPlanCommand(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("plan", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	manifest := fs.String("matrix-manifest", "", "full Cartesian matrix manifest JSON")
	output := fs.String("out", "-", "write the pairwise plan JSON to this file, or - for stdout")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *manifest == "" {
		return errors.New("usage: llama-bench-harness plan --matrix-manifest matrix.json [--out plan.json]")
	}
	return writePairwisePlan(*manifest, *output, stdout)
}

// coverageCell is one explicitly tested setting/thread combination in a
// pairwise plan. It covers every pair of values across all manifest axes.
type coverageCell struct {
	Settings analysisSettings `json:"settings"`
	Threads  int              `json:"threads"`
}

type coverageFactor struct {
	name   string
	values []string
	apply  func(*analysisSettings, string)
}

func coverageFactors(manifest analysisMatrixManifest) ([]coverageFactor, error) {
	factors := []coverageFactor{
		{name: "models", values: manifest.Models, apply: func(s *analysisSettings, v string) { s.Model = v }},
		{name: "contexts", values: intsToStrings(manifest.Contexts), apply: func(s *analysisSettings, v string) { s.Context, _ = strconv.Atoi(v) }},
		{name: "parallel", values: intsToStrings(manifest.Parallel), apply: func(s *analysisSettings, v string) { s.Parallel, _ = strconv.Atoi(v) }},
		{name: "batches", values: intsToStrings(manifest.Batches), apply: func(s *analysisSettings, v string) { s.Batch, _ = strconv.Atoi(v) }},
		{name: "ubatches", values: intsToStrings(manifest.UBatches), apply: func(s *analysisSettings, v string) { s.UBatch, _ = strconv.Atoi(v) }},
		{name: "fit_targets", values: intsToStrings(manifest.FitTargets), apply: func(s *analysisSettings, v string) { s.FitTarget, _ = strconv.Atoi(v) }},
		{name: "cache_reuse", values: intsToStrings(manifest.CacheReuse), apply: func(s *analysisSettings, v string) { s.CacheReuse, _ = strconv.Atoi(v) }},
		{name: "flash_attention", values: manifest.FlashAttention, apply: func(s *analysisSettings, v string) { s.FlashAttention = v }},
		{name: "unified_kv", values: boolsToStrings(manifest.UnifiedKV), apply: func(s *analysisSettings, v string) { s.UnifiedKV, _ = strconv.ParseBool(v) }},
		{name: "kv_per_slot", values: intsToStrings(manifest.KVPerSlot), apply: func(s *analysisSettings, v string) { s.KVPerSlot, _ = strconv.Atoi(v) }},
		{name: "kv_k", values: manifest.KVK, apply: func(s *analysisSettings, v string) { s.KVK = v }},
		{name: "kv_v", values: manifest.KVV, apply: func(s *analysisSettings, v string) { s.KVV = v }},
		{name: "preserve_thinking", values: boolsToStrings(manifest.PreserveThinking), apply: func(s *analysisSettings, v string) { s.PreserveThinking, _ = strconv.ParseBool(v) }},
		{name: "threads", values: intsToStrings(manifest.Threads)},
	}
	for _, factor := range factors {
		if len(factor.values) == 0 {
			return nil, fmt.Errorf("matrix manifest: axis %s has no values", factor.name)
		}
		seen := make(map[string]bool, len(factor.values))
		for _, value := range factor.values {
			if seen[value] {
				return nil, fmt.Errorf("matrix manifest: axis %s repeats value %q", factor.name, value)
			}
			seen[value] = true
		}
	}
	return factors, nil
}

func intsToStrings(values []int) []string {
	out := make([]string, len(values))
	for i, value := range values {
		out[i] = strconv.Itoa(value)
	}
	return out
}

func boolsToStrings(values []bool) []string {
	out := make([]string, len(values))
	for i, value := range values {
		out[i] = strconv.FormatBool(value)
	}
	return out
}

func generatePairwisePlan(manifest analysisMatrixManifest) ([]coverageCell, error) {
	factors, err := coverageFactors(manifest)
	if err != nil {
		return nil, err
	}
	candidates := buildPairwiseCandidates(factors)
	return selectPairwiseCells(candidates, factors)
}

type pairwiseCandidate struct {
	cell   coverageCell
	values []string
}

func buildPairwiseCandidates(factors []coverageFactor) []pairwiseCandidate {
	candidates := []pairwiseCandidate{{values: make([]string, len(factors))}}
	for i, factor := range factors {
		next := make([]pairwiseCandidate, 0, len(candidates)*len(factor.values))
		for _, partial := range candidates {
			for _, value := range factor.values {
				copyValues := append([]string(nil), partial.values...)
				copyValues[i] = value
				candidate := pairwiseCandidate{cell: partial.cell, values: copyValues}
				if factor.name == "threads" {
					candidate.cell.Threads, _ = strconv.Atoi(value)
				} else {
					factor.apply(&candidate.cell.Settings, value)
				}
				next = append(next, candidate)
			}
		}
		candidates = next
	}
	return candidates
}

func selectPairwiseCells(candidates []pairwiseCandidate, factors []coverageFactor) ([]coverageCell, error) {
	pairs := allPairwiseKeys(candidates, len(factors))
	remaining := len(pairs)
	selected := make([]bool, len(candidates))
	var plan []coverageCell
	for remaining > 0 {
		bestIndex := highestScoringCandidate(candidates, factors, selected, pairs)
		if bestIndex < 0 {
			return nil, errors.New("pairwise planner could not cover all factor pairs")
		}
		selected[bestIndex] = true
		best := candidates[bestIndex]
		plan = append(plan, best.cell)
		remaining -= removeCoveredPairs(pairs, best.values, len(factors))
	}
	return plan, nil
}

func allPairwiseKeys(candidates []pairwiseCandidate, factorCount int) map[string]bool {
	pairs := make(map[string]bool)
	for _, candidate := range candidates {
		for i := 0; i < factorCount; i++ {
			for j := i + 1; j < factorCount; j++ {
				pairs[pairKey(i, candidate.values[i], j, candidate.values[j])] = true
			}
		}
	}
	return pairs
}

func highestScoringCandidate(candidates []pairwiseCandidate, factors []coverageFactor, selected []bool, pairs map[string]bool) int {
	bestIndex, bestScore := -1, 0
	for i, candidate := range candidates {
		if selected[i] {
			continue
		}
		score := candidatePairScore(candidate.values, len(factors), pairs)
		if score > bestScore {
			bestIndex, bestScore = i, score
		}
	}
	return bestIndex
}

func candidatePairScore(values []string, factorCount int, pairs map[string]bool) int {
	score := 0
	for a := 0; a < factorCount; a++ {
		for b := a + 1; b < factorCount; b++ {
			if pairs[pairKey(a, values[a], b, values[b])] {
				score++
			}
		}
	}
	return score
}

func removeCoveredPairs(pairs map[string]bool, values []string, factorCount int) int {
	removed := 0
	for a := 0; a < factorCount; a++ {
		for b := a + 1; b < factorCount; b++ {
			key := pairKey(a, values[a], b, values[b])
			if pairs[key] {
				delete(pairs, key)
				removed++
			}
		}
	}
	return removed
}

func pairKey(a int, av string, b int, bv string) string {
	return strconv.Itoa(a) + ":" + av + "|" + strconv.Itoa(b) + ":" + bv
}

func writePairwisePlan(manifestPath, outputPath string, stdout io.Writer) error {
	manifest, err := readAnalysisMatrixManifest(manifestPath)
	if err != nil {
		return err
	}
	plan, err := generatePairwisePlan(manifest)
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(plan, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if outputPath == "" || outputPath == "-" {
		_, err = stdout.Write(data)
		return err
	}
	if err := os.WriteFile(outputPath, data, 0o644); err != nil {
		return err
	}
	_, err = fmt.Fprintf(stdout, "Pairwise plan: %s (%d setting/thread cells)\n", outputPath, len(plan))
	return err
}
