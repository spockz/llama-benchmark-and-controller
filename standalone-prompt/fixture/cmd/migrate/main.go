// Command migrate demonstrates the established service dispatch path.
package main

import (
	"os"

	"example.com/auditapp/audit"
	"example.com/auditapp/service"
)

func main() {
	dispatcher := service.NewDispatcher(audit.NewMemorySink())
	_ = dispatcher.Record(audit.Event{ID: "startup", Kind: os.Getenv("AUDIT_KIND")})
}
