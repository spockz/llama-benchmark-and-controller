// Package audit exposes the established event sink contract and a memory implementation.
package audit

type Sink interface {
	Write(Event) error
}

type TransactionalSink interface {
	Sink
	Begin() error
	Commit() error
}

type MemorySink struct {
	Events []Event
}

func NewMemorySink() *MemorySink {
	return &MemorySink{}
}

func (m *MemorySink) Write(event Event) error {
	event.Kind = Normalize(event.Kind)
	m.Events = append(m.Events, event)
	return nil
}
