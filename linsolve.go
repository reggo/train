package train

import (
	"errors"

	"github.com/reggo/common"
	"github.com/reggo/regularize"

	"github.com/gonum/matrix/mat64"
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
type Linear interface {
	// NumFeatures returns the number of features
	NumFeatures() int

	// Featurize transforms the input into the elements of the feature matrix. Feature
	// will have length NumFeatures()
	Featurize(input, feature []float64)

	// CanParallelize returns true if Featurize can be called in parallel
	CanParallelize() bool
}

// IsLinearRegularizer returns true if the regularizer can be used with LinearSolve
func IsLinearRegularizer(r regularize.Regularizer) bool {
	switch r.(type) {
	case regularize.None:
		return true
	default:
		return false
	}
}

// LinearSolve trains a Linear algorithm. Will return nil if inputs and outputs don't have the same number of rows and if
func LinearSolve(l Linear, inputs, trueOutputs *mat64.Dense, weights []float64, r regularize.Regularizer) (parameters *mat64.Dense, err error) {
	if !IsLinearRegularizer(r) {
		return nil, errors.New("Regularizer type not supported")
	}

	weights, err = verifyInputs(inputs, trueOutputs, weights)
	if err != nil {
		return nil, err
	}

	nSamples, nInputs := inputs.Dims()
	nFeatures := l.NumFeatures()

	// Create the memory for storing the results from featurize
	feature := make([]float64, nFeatures*nSamples)

	parallel := l.CanParallelize()

	input := make([]float64, nInputs)
	f := func(start, end int) {
		for i := start; i < end; i++ {
			inputs.Row(input, i)
			l.Featurize(input, feature[i*nSamples:(i+1)*nSamples])
		}
	}
	if parallel {
		parallelFor(nSamples, getGrain(nSamples), f)
	} else {
		f(0, nSamples)
	}

	A := mat64.NewDense(nSamples, nFeatures, feature)

	switch r.(type) {
	case regularize.None:
	default:
		panic("Shouldn't be here. Must be error in IsLinearRegularizer")
	}
	parameters = mat64.Solve(A, trueOutputs)
	return parameters, nil
}
