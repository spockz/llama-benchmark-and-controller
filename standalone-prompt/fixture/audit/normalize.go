// Package audit normalizes event kinds at the delivery boundary.
package audit

import "strings"

// Normalize trims whitespace and lowercases an event kind so that
// downstream classification and persistence see a canonical label.
func Normalize(kind string) string {
	return strings.ToLower(strings.TrimSpace(kind))
}
