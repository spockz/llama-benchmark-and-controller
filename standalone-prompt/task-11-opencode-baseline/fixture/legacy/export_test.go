// Package legacy protects its exporter and classification from audit changes.
package legacy

import (
	"bytes"
	"testing"
)

func TestExporterAndClassification(t *testing.T) {
	var output bytes.Buffer
	if err := (Exporter{Output: &output}).Write([]byte("archive")); err != nil {
		t.Fatalf("write export: %v", err)
	}
	if output.String() != "archive" || Classify("probe") != "archive" {
		t.Fatal("legacy protocol changed unexpectedly")
	}
}
