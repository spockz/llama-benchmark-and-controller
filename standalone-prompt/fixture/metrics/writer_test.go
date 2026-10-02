// Package metrics protects its writer and classification protocols from audit changes.
package metrics

import "testing"

func TestWriterAndClassification(t *testing.T) {
	writer := &Writer{}
	writer.Write(3)
	if writer.Total != 3 || Classify("probe") != "sample" {
		t.Fatalf("metrics protocol changed unexpectedly: %#v", writer)
	}
}
