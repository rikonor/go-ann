package ann

import (
	mat "github.com/gonum/matrix/mat64"
	"github.com/rikonor/go-ann/ksmall"
)

type exhaustive struct {
	xs [][]float64
}

// NewExhaustiveNNer creates a new ANNer that uses exhaustive search
// Obviously you should use this for all your performance sensitive tasks
func NewExhaustiveNNer(xs [][]float64) ANNer {
	return &exhaustive{xs: xs}
}

func (nn *exhaustive) ANN(q []float64, k int) []int {
	if len(nn.xs) == 0 {
		return []int{}
	}

	n, d := len(nn.xs), len(nn.xs[0])

	// Put the query value into a vector
	qVec := mat.NewVector(len(q), q)

	// Put our data into a matrix
	X := mat.NewDense(d, n, nil)
	for i := 0; i < n; i++ {
		X.SetCol(i, nn.xs[i])
	}

	// Calculate distances
	distsVec := mat.NewVector(n, nil)
	for i := 0; i < n; i++ {
		distVec := mat.NewVector(d, nil)
		distVec.SubVec(
			X.ColView(i),
			qVec,
		)
		dist := mat.Norm(distVec, 2)
		distsVec.SetVec(i, dist)
	}

	// Get k closest results
	distances := mat.Col(nil, 0, distsVec)
	return ksmall.KSmallestIndices(distances, k)
}
