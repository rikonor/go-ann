package ann

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/kr/pretty"
)

func TestExhaustiveSearch(t *testing.T) {
	nn := NewExhaustiveNNer([][]float64{
		[]float64{0, 0, 0},
		[]float64{1.1, 1.1, 1.1},
		[]float64{2, 2, 2},
	})

	p := nn.NN([]float64{1, 1, 1})

	if !reflect.DeepEqual(p, []float64{1.1, 1.1, 1.1}) {
		t.Fatalf("Found wrong nearest neighbor: %v", p)
	}
}

func TestMRPTNNer(t *testing.T) {
	nn := NewMRPTNNer(1, 3, [][]float64{
		[]float64{0, 0, 0, 0},
		[]float64{1.1, 1.1, 1.1, 1.1},
		[]float64{2, 2, 2, 2},
	})

	r := mat64.NewDense(4, 3, nil)
	r.SetCol(0, []float64{1, 0, 0, 0})
	r.SetCol(1, []float64{1, 1, 0, -1})
	r.SetCol(2, []float64{1, -1, 1, 0})

	nnmrpt := nn.(*mrpt)
	nnmrpt.trees = []*tree{
		&tree{
			r: r,
			root: &node{
				split: 1,
				left: &node{
					split: 1,
					left: &node{
						xs: [][]float64{[]float64{0, 0, 0, 0}},
					},
					right: &node{
						xs: [][]float64{[]float64{1, 1, 1, 1}},
					},
				},
				right: &node{
					split: 1,
					left: &node{
						xs: [][]float64{[]float64{2, 2, 2, 2}},
					},
					right: &node{
						xs: [][]float64{[]float64{3, 3, 3, 3}},
					},
				},
			},
		},
	}

	p := nn.NN([]float64{1, 1, 1, 1})

	fmt.Println(p)

	fmt.Printf("%# v\n", pretty.Formatter(nn))
}
