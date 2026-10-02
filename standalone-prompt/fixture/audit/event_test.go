// Package audit verifies stable event data independent of delivery rules.
package audit

import "testing"

func TestEventRetainsKind(t *testing.T) {
	event := Event{ID: "evt-1", Kind: "audit"}
	if event.ID != "evt-1" || event.Kind != "audit" {
		t.Fatalf("event = %#v", event)
	}
}
