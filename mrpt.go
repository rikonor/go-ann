package ann

// type mrpt struct {
// 	trees []*tree
// }
//
// type tree struct {
// 	root *node
// 	r    mat64.Matrix
// }
//
// type node struct {
// 	split float64
// 	left  *node
// 	right *node
//
// 	xs [][]float64
// }
//
// // NewMRPTNNer creates a NN index using random projection trees
// // See https://arxiv.org/pdf/1509.06957.pdf for additional details
// // t -> number of trees, l -> depth of tree
// func NewMRPTNNer(t int, l int, xs [][]float64) NNer {
// 	a := 0.5 // TODO(temporary)
// 	return &mrpt{trees: growTrees(xs, t, l, a)}
// }
//
// func growTrees(xs [][]float64, t int, l int, a float64) []*tree {
// 	// Number of vectors
// 	n := len(xs)
//
// 	// Infer vector dimension from xs
// 	d := len(xs[0])
//
// 	// Convert xs to a matrix
// 	// TODO should remove this X thing
// 	X := mat64.NewDense(d, n, nil)
// 	for i := 0; i < n; i++ {
// 		X.SetCol(i, xs[i])
// 	}
// 	fmt.Printf("Vectors:\n%v\n", mat64.Formatted(X))
//
// 	fmt.Printf("Starting to grow %d trees of depth %d for %d vectors [a=%f]:\n",
// 		t, len(xs), l, a)
//
// 	trees := []*tree{}
//
// 	for i := 0; i < t; i++ {
// 		fmt.Printf("Building a random projection matrix with %d %d-dimensional vectors\n",
// 			l-1, d)
//
// 		// Create a new random projection matrix
// 		r := mat64.NewDense(d, l-1, nil)
// 		// Create one random vector per tree level
// 		for j := 0; j < (l - 1); j++ {
// 			vs := []float64{}
//
// 			for k := 0; k < d; k++ {
// 				// TODO: Use a sparse vector strategy
// 				vs = append(vs, rand.NormFloat64())
// 			}
//
// 			// Set the random vector into the matrix
// 			r.SetCol(j, vs)
// 		}
//
// 		fmt.Printf("RP Matrix #%d:\n%v\n",
// 			i, mat64.Formatted(r))
//
// 		// Create a new tree
// 		trees = append(trees, &tree{
// 			r:    r,
// 			root: growTree(xs, l, 0, r),
// 		})
// 	}
//
// 	return trees
// }
//
// // growTree is a recursive function for building a RP tree
// // xs -> points
// // r -> random projection matrix
// func growTree(xs [][]float64, l, level int, r *mat64.Dense) *node {
// 	fmt.Printf("\nBuilding tree level %d\n", level)
// 	if level == l-1 {
// 		fmt.Printf("Returning leaf with vectors: %v\n", xs)
// 		return &node{xs: xs}
// 	}
//
// 	// Get the random projection vector of the current level
// 	rv := r.ColView(level)
//
// 	fmt.Printf("Calculating projections using RP vector:\n%v\n", mat64.Formatted(rv))
// 	projVals := []float64{}
// 	for _, x := range xs {
// 		// Convert x to a vector
// 		xv := mat64.NewVector(len(x), x)
//
// 		// Get projection
// 		projVals = append(projVals, mat64.Dot(xv, rv))
// 	}
// 	fmt.Println("Projections for this RP vector:", projVals)
//
// 	// Get the median of the projections
// 	split := median(projVals)
//
// 	// Put vectors in left or right subtree
// 	// based on which side of the split their project falls
// 	xsLeft := [][]float64{}
// 	xsRight := [][]float64{}
//
// 	for i, proj := range projVals {
// 		fmt.Printf("Projection for %v: %f\n", xs[i], proj)
// 		if proj <= split {
// 			xsLeft = append(xsLeft, xs[i])
// 		} else {
// 			xsRight = append(xsRight, xs[i])
// 		}
// 	}
// 	fmt.Printf("Left node: %v\n", xsLeft)
// 	fmt.Printf("Right node: %v\n", xsRight)
//
// 	return &node{
// 		split: split,
// 		left:  growTree(xsLeft, l, level+1, r),
// 		right: growTree(xsRight, l, level+1, r),
// 	}
// }
//
// func (nn *mrpt) NN(q []float64) []float64 {
// 	// Keep candidates in a set
// 	xsSet := map[string][]float64{}
// 	votes := map[string]int{}
//
// 	// How many votes does a vector need to be included in the output set
// 	reqVotes := 1
//
// 	// Query the trees to get candidates
// 	for _, tree := range nn.trees {
// 		xs := queryTree(tree, q)
// 		for _, x := range xs {
// 			// Count vote
// 			k := fmt.Sprintf("%v", x)
// 			votes[k]++
//
// 			// If vector has enough votes, include in output set
// 			if votes[k] == reqVotes {
// 				xsSet[k] = x
// 			}
// 		}
// 	}
//
// 	xsCandidates := [][]float64{}
// 	for _, x := range xsSet {
// 		xsCandidates = append(xsCandidates, x)
// 	}
//
// 	// Perform naive k-nearest-neighbor search on candidates set
// 	knn := NewExhaustiveNNer(xsCandidates)
// 	return knn.NN(q)
// }
//
// func queryTree(tree *tree, q []float64) [][]float64 {
// 	fmt.Printf("\nLooking for NN for %v\n", q)
//
// 	// Get vector dimension and depth of tree
// 	d, l := tree.r.Dims()
//
// 	// Convert q to a vector so we can perform matrix math
// 	qv := mat64.NewVector(d, q)
//
// 	// Project query point onto tree's random matrix
// 	var p mat64.Vector
// 	p.MulVec(tree.r.T(), qv)
//
// 	// Traverse the tree until point lands in a bucket
// 	node := tree.root
// 	for i := 0; i < l; i++ {
// 		if p.At(i, 0) <= node.split {
// 			node = node.left
// 		} else {
// 			node = node.right
// 		}
// 	}
//
// 	return node.xs
// }
//
// // median calculates the median value of a series of elements
// func median(vals []float64) float64 {
// 	// Make a copy so we don't alter the given slice
// 	vs := make([]float64, len(vals))
// 	copy(vs, vals)
//
// 	sort.Float64s(vs)
//
// 	// Even number of elements
// 	if len(vs)%2 == 0 {
// 		m := (vs[len(vs)/2-1] + vs[len(vs)/2]) / 2
// 		return m
// 	}
//
// 	// Odd number of elements
// 	m := vs[len(vs)/2]
// 	return m
// }
