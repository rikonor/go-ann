go-ann
---

[![GoDoc](https://godoc.org/github.com/rikonor/go-ann?status.svg)](http://godoc.org/github.com/rikonor/go-ann)
[![Build
Status](https://travis-ci.org/rikonor/go-ann.svg?branch=master)](https://travis-ci.org/rikonor/go-ann)

Pure Go implementation of Approximate k-Nearest-Neighbor search.

This package exposes:
- A naive exact match implementation which can be used for testing purposes (`ExhaustiveANNer`).
- An Approximate search based on the [MRPT algorithm](https://arxiv.org/pdf/1509.06957.pdf) (`MRPTANNer`).

### Usage

The package exposes an `ANNer` interface:
```go
// ANNer allows you to perform an approximate k-NN search given a query point
type ANNer interface {
	// ANN takes a query point and how many nearest neighbors to return
	// and returns the indices of the neihgbors
	ANN(q []float64, k int) []int
}
```

***Basic example***

```go
// Assuming we have a series of vectors xs
xs := [][]float64{
  []float64{0, 0, 0, 0},
  []float64{1.1, 1.1, 1.1, 1.1},
  []float64{2, 2, 2, 2},
  ...
}

// Decide on a query point q
q := []float64{1, 1, 1, 1}

// We'd like to find the index of it's nearest neighbor
k := 1
```

Using exact/Exhaustive search
```go
// Create a naive NN index using exact/exhaustive search
nn := NewExhaustiveNNer(xs)

indices := nn.ANN(q, k)
// indices -> [1] for (1.1, 1.1, 1.1, 1.1)
```

Using approximate search

```go
// Notice that the MRPT algorithm has a few tunable parameters: Number of trees and tree depth

// Create a new ANN index using the MRPT algorithm
nn := NewMRPTANNer(
  3,  // Number of trees
  10, // Tree depth
  xs, // Vectors
)

indices := nn.ANN(q, k)
// indices -> [1] for (1.1, 1.1, 1.1, 1.1)
```

***Using a MappedANNer***

Wrap an ANNer with a MappedANNer to associate values with vectors.

```go
// vectorIDs are values which are associated with their respective vectors
// e.g We'd like to associate vector #0 with "1", etc
vectorIDs := []string{"1", "2", "3", "4", ...}

mnn := NewMappedANNer(nn, vectorIDs)

q := []float64{1, 2, 3}
k := 2

// Use as usual, however, now the return value is the associated value
// rather then the vector indices
vs := mnn.ANN(q, k)
```

***Very basic benchmark***

```
BenchmarkMRPTANNer-8               10000            138771 ns/op
BenchmarkExhaustiveNNer-8            100          16952058 ns/op
```

### License

MIT
