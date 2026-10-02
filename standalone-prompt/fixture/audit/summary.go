// Package audit groups delivery kinds for operational reporting.
package audit

func Classify(kind string) string {
	kind = Normalize(kind)
	switch kind {
	case "health", "probe":
		return "control"
	default:
		return "data"
	}
}
