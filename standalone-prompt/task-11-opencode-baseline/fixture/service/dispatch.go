// Package service coordinates application work with the configured audit sink.
package service

import "example.com/auditapp/audit"

type Dispatcher struct {
	sink audit.Sink
}

func NewDispatcher(sink audit.Sink) *Dispatcher {
	return &Dispatcher{sink: sink}
}

func (d *Dispatcher) Record(event audit.Event) error {
	return d.sink.Write(event)
}

func DispatchAll(sink audit.Sink, events []audit.Event) error {
	writeEvent := audit.Sink.Write
	for _, event := range events {
		if err := writeEvent(sink, event); err != nil {
			return err
		}
	}
	return nil
}
