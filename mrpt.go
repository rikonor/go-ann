package ann

import (
	"math/rand"
	"sort"

	mat "github.com/gonum/matrix/mat64"
)

type mrpt struct {
	xs    [][]float64
	trees []*tree
}

type tree struct {
	root *node
	r    mat.Matrix
}

type node struct {
	split float64
	left  *node
	right *node

	indices []int
}

// NewMRPTANNer creates a NN index using random projection trees
// See https://arxiv.org/pdf/1509.06957.pdf for additional details
// t -> number of trees, l -> depth of tree
func NewMRPTANNer(t int, l int, xs [][]float64) ANNer {
	a := 0.5 // TODO(temporary)

	// Infer dimensions from input matrix
	n, d := len(xs), len(xs[0])

	// Convert xs to a gonum matrix
	X := mat.NewDense(d, n, nil)
	for i := 0; i < n; i++ {
		X.SetCol(i, xs[i])
	}

	return &mrpt{
		xs:    xs,
		trees: growTrees(X, t, l, a),
	}
}

func growTrees(X mat.Matrix, t, l int, a float64) []*tree {
	// feature vector dimension
	d, n := X.Dims()

	trees := []*tree{}

	// Create t trees
	for i := 0; i < t; i++ {
		// Create a RP matrix
		R := mat.NewDense(d, l-1, nil)
		// Create one random vector per tree level
		for j := 0; j < (l - 1); j++ {
			rs := []float64{}

			for k := 0; k < d; k++ {
				// TODO: only use non-zero value with prob a

				// TODO: Use a sparse vector strategy
				rs = append(rs, rand.NormFloat64())
			}

			// Set the random vector into the matrix
			R.SetCol(j, rs)
		}

		// Calculate the projections
		var P mat.Dense
		P.Mul(X.T(), R)

		// Create indices list
		indices := []int{}
		for i := 0; i < n; i++ {
			indices = append(indices, i)
		}

		trees = append(trees, &tree{
			r:    R,
			root: growTree(&P, l, 0, indices),
		})
	}

	return trees
}

func growTree(P mat.Matrix, l, level int, indices []int) *node {
	// Stop if we're at the leaf
	if level == l-1 {
		return &node{indices: indices}
	}

	// Get the projections for this level
	ps := mat.Col(nil, level, P)

	// Get the median of the projections
	m := median(ps)

	// Divide indices to left and right based on median value
	leftIndices := []int{}
	rightIndices := []int{}
	for _, i := range indices {
		if ps[i] <= m {
			leftIndices = append(leftIndices, i)
		} else {
			rightIndices = append(rightIndices, i)
		}
	}

	return &node{
		left:  growTree(P, l, level+1, leftIndices),
		right: growTree(P, l, level+1, rightIndices),
		split: m,
	}
}

func (nn *mrpt) ANN(q []float64, k int) []int {
	// Keep track of votes
	votesMap := map[int]int{}

	// How many votes does a vector need to be included in the output set
	reqVotes := 1

	// Query the trees to get candidates
	for _, tree := range nn.trees {
		indices := queryTree(tree, q, k)
		for _, i := range indices {
			// Count vote
			votesMap[i]++
		}
	}

	xsCandidates := [][]float64{}
	xsIndices := []int{}
	for i, votes := range votesMap {
		if votes >= reqVotes {
			xsCandidates = append(xsCandidates, nn.xs[i])
			// Track the index of each vector so we can retrieve it later
			xsIndices = append(xsIndices, i)
		}
	}

	// Perform naive k-nearest-neighbor search on candidates set
	knn := NewExhaustiveNNer(xsCandidates)
	knnIndices := knn.ANN(q, k)

	// Convert the above knnIndices to the indices of the vectors in the scope
	// of all of our data
	indices := []int{}
	for _, i := range knnIndices {
		indices = append(indices, xsIndices[i])
	}

	return indices
}

func queryTree(tree *tree, q []float64, k int) []int {
	return []int{}
	// fmt.Printf("\nLooking for NN for %v\n", q)
	//
	// // Get vector dimension and depth of tree
	// d, l := tree.r.Dims()
	//
	// // Convert q to a vector so we can perform matrix math
	// qv := mat64.NewVector(d, q)
	//
	// // Project query point onto tree's random matrix
	// var p mat64.Vector
	// p.MulVec(tree.r.T(), qv)
	//
	// // Traverse the tree until point lands in a bucket
	// node := tree.root
	// for i := 0; i < l; i++ {
	// 	if p.At(i, 0) <= node.split {
	// 		node = node.left
	// 	} else {
	// 		node = node.right
	// 	}
	// }
	//
	// return node.xs
}

// func querySubtree(n *node, q []float64, k int) []int {
//
// }

// median calculates the median value of a series of elements
func median(vals []float64) float64 {
	// Make a copy so we don't alter the given slice
	vs := make([]float64, len(vals))
	copy(vs, vals)

	sort.Float64s(vs)

	// Even number of elements
	if len(vs)%2 == 0 {
		m := (vs[len(vs)/2-1] + vs[len(vs)/2]) / 2
		return m
	}

	// Odd number of elements
	m := vs[len(vs)/2]
	return m
}
