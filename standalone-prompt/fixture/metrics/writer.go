// Package metrics owns unrelated classification and writer protocols.
package metrics

type Writer struct {
	Total int
}

func (w *Writer) Write(delta int) {
	w.Total += delta
}

func Classify(name string) string {
	if name == "probe" {
		return "sample"
	}
	return "measurement"
}
