// Package audit groups delivery kinds for operational reporting.
package audit

func Classify(kind string) string {
	switch kind {
	case "health":
		return "control"
	default:
		return "data"
	}
}
