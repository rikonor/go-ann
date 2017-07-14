package ann

// NNer ...
type NNer interface {
	NN(q []float64) []float64
}

type exhaustive struct {
	xs [][]float64
}

// NewExhaustiveNNer creates a new NNer that uses exhaustive search
// Obviously you should use this for all your performance sensitive tasks
func NewExhaustiveNNer(xs [][]float64) NNer {
	return &exhaustive{xs: xs}
}

func (nn *exhaustive) NN(q []float64) []float64 {
	pnn := nn.xs[0]
	dnn := euclideanDistanceSqrd(q, pnn)

	for _, x := range nn.xs[1:] {
		d := euclideanDistanceSqrd(q, x)
		if d < dnn {
			pnn = x
			dnn = d
		}
	}

	return pnn
}

type kdtree struct{}
