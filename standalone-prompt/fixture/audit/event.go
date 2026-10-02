// Package audit defines the event delivery boundary used by the fixture.
package audit

type Event struct {
	ID   string `json:"id"`
	Kind string `json:"kind"`
}
