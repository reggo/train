package train

import (
	"errors"

	"github.com/reggo/common"
	"github.com/reggo/loss"
	"github.com/reggo/regularize"

	"github.com/gonum/matrix/mat64"
)

const (
	minGrain = 1
	maxGrain = 500
)

func verifyInputs(inputs, outputs *mat64.Dense, weights []float64) ([]float64, error) {
	// TODO: Replace this
	nSamples, _ := inputs.Dims()
	nOutputSamples, _ := outputs.Dims()
	if nSamples != nOutputSamples {
		return weights, common.DataMismatch{Input: nSamples, Output: nOutputSamples, Weight: len(weights)}
	}
	if len(weights) != 0 {
		nWeightSamples := len(weights)
		if nWeightSamples != nSamples {
			return weights, common.DataMismatch{Input: nSamples, Output: nOutputSamples, Weight: len(weights)}
		}
	}
	weights = make([]float64, nSamples)
	for i := range weights {
		weights[i] = 1
	}
	return weights, nil
}

// Linear is a type whose parameters are a linear combination of a set of features
type LinearSolver interface {
	// NumFeatures returns the number of features
	NumFeatures() int

	// Featurize transforms the input into the elements of the feature matrix. Feature
	// will have length NumFeatures()
	Featurize(input, feature []float64)

	// CanParallelize returns true if Featurize can be called in parallel
	CanParallelize() bool
}

// IsLinearRegularizer returns true if the regularizer can be used with LinearSolve
func IsLinearSolveRegularizer(r regularize.Regularizer) bool {
	switch r.(type) {
	case regularize.None:
		return true
	default:
		return false
	}
}

func IsLinearSolveLosser(l loss.Losser) bool {
	switch l.(type) {
	case loss.SquaredDistance:
		return true
	default:
		return false
	}
}

// LinearSolve trains a Linear algorithm.
// Assumes inputs and outputs are already scaled
func LinearSolve(l LinearSolver, inputs, trueOutputs *mat64.Dense, weights []float64, r regularize.Regularizer) (parameters *mat64.Dense, err error) {
	if !IsLinearSolveRegularizer(r) {
		return nil, errors.New("Regularizer type not supported")
	}

	weights, err = verifyInputs(inputs, trueOutputs, weights)
	if err != nil {
		return nil, err
	}

	nSamples, _ := inputs.Dims()
	nFeatures := l.NumFeatures()

	// Create the memory for storing the results from featurize
	//feature := make([]float64, nFeatures*nSamples)
	feature := mat64.NewDense(nSamples, nFeatures, nil)

	parallel := l.CanParallelize()
	f := func(start, end int) {
		for i := start; i < end; i++ {
			l.Featurize(inputs.RowView(i), feature.RowView(i))
		}
	}
	if parallel {
		common.ParallelFor(nSamples, common.GetGrainSize(nSamples, minGrain, maxGrain), f)
	} else {
		f(0, nSamples)
	}

	A := feature

	switch r.(type) {
	case regularize.None:
	default:
		panic("Shouldn't be here. Must be error in IsLinearRegularizer")
	}
	parameters = mat64.Solve(A, trueOutputs)
	return parameters, nil
}
