package train

import (
	"runtime"
	"sync"
	"sync/atomic"
)

const (
	minGrainSize = 1
	maxGrainSize = 500
)

func getGrain(nSamples int) int {
	procs := runtime.GOMAXPROCS(0)
	grainPerProc := nSamples / procs
	if grainPerProc < minGrainSize {
		return minGrainSize
	}
	if grainPerProc > maxGrainSize {
		return maxGrainSize
	}
	return grainPerProc
}

func parallelFor(n, grain int, f func(start, end int)) {
	P := runtime.GOMAXPROCS(0)
	idx := uint64(0)
	var wg sync.WaitGroup
	wg.Add(P)
	for p := 0; p < P; p++ {
		go func() {
			for {
				start := int(atomic.AddUint64(&idx, uint64(grain))) - grain
				if start >= n {
					break
				}
				end := start + grain
				if end > n {
					end = n
				}
				f(start, end)
			}
			wg.Done()
		}()
	}
	wg.Wait()
}
