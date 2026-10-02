// Package audit verifies stable reporting categories.
package audit

import "testing"

func TestClassifyHealth(t *testing.T) {
	if got := Classify("health"); got != "control" {
		t.Fatalf("classification = %q, want control", got)
	}
}
