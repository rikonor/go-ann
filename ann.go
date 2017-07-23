package ann

// ANNer allows you to perform an approximate k-NN search given a query point
type ANNer interface {
	// ANN takes a query point and how many nearest neighbors to return
	// and returns the indices of the neihgbors
	ANN(q []float64, k int) []int
}
