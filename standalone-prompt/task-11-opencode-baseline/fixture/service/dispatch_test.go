// Package service verifies ordinary event delivery.
package service

import (
	"testing"

	"example.com/auditapp/audit"
)

func TestRecordDeliversEvent(t *testing.T) {
	sink := audit.NewMemorySink()
	dispatcher := NewDispatcher(sink)
	if err := dispatcher.Record(audit.Event{ID: "evt-1", Kind: "audit"}); err != nil {
		t.Fatalf("record event: %v", err)
	}
	if len(sink.Events) != 1 || sink.Events[0].Kind != "audit" {
		t.Fatalf("events = %#v, want delivered audit event", sink.Events)
	}
}
