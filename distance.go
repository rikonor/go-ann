package ann

// euclideanDistanceSqrd returns the squared euclidaen distance of two points
func euclideanDistanceSqrd(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("ann: length mismatch")
	}

	var s float64
	for i := 0; i < len(a); i++ {
		s += (a[i] - b[i]) * (a[i] - b[i])
	}
	return s
}
