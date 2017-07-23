package ann

// func TestExhaustiveSearch(t *testing.T) {
// 	nn := NewExhaustiveNNer([][]float64{
// 		[]float64{0, 0, 0},
// 		[]float64{1.1, 1.1, 1.1},
// 		[]float64{2, 2, 2},
// 	})
//
// 	p := nn.NN([]float64{1, 1, 1})
//
// 	if !reflect.DeepEqual(p, []float64{1.1, 1.1, 1.1}) {
// 		t.Fatalf("Found wrong nearest neighbor: %v", p)
// 	}
// }
//
// func TestMRPTNNerNew(t *testing.T) {
// 	nn := NewMRPTNNer(1, 3, [][]float64{
// 		[]float64{1.1, 1.1, 1.1, 1.1},
// 		[]float64{0, 0, 0, 0},
// 		[]float64{2, 2, 2, 2},
// 		[]float64{3, 3, 3, 3},
// 		[]float64{4, 4, 4, 4},
// 		[]float64{5, 5, 5, 5},
// 	})
//
// 	q := []float64{2.9, 2.9, 2.9, 2.9}
// 	fmt.Printf("Querying for %v\n", q)
// 	p := nn.NN(q)
// 	fmt.Printf("\nResult: %v\n", p)
// }
