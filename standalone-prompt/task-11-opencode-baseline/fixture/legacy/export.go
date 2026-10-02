// Package legacy retains byte export and archival classifications outside audit delivery.
package legacy

import "io"

type Exporter struct {
	Output io.Writer
}

func (e Exporter) Write(data []byte) error {
	_, err := e.Output.Write(data)
	return err
}

func Classify(kind string) string {
	if kind == "probe" {
		return "archive"
	}
	return "record"
}
