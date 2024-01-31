package main

import (
	"encoding/csv"
	"fmt"
	"image"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

func Plot(ps ...plot.Plotter) *image.RGBA {
	p := plot.New()
	p.Add(append([]plot.Plotter{
		plotter.NewGrid(),
	}, ps...)...)
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	c := vgimg.NewWith(vgimg.UseImage(img))
	p.Draw(draw.New(c))
	return c.Image().(*image.RGBA)
}

func main() {
	rand := rand.New(rand.NewSource(time.Now().UnixNano()))
	ebiten.SetWindowSize(640, 480)
	ebiten.SetWindowTitle("Gradient descent")

	const (
		epochs              = 2000
		printEveryNthEpochs = 100
		learningRateW       = 0.5e-4
		learningRateB       = 0.7

		plotLoss = false // Loss curve: true, Resulting line: false.

		//inputPoints                      = 25
		inputPointsMinX, inputPointsMaxX = 0, 150
		//inputPointsRandY                 = 1 // Makes sure Ys aren't on the line, but around it. Randomly.
		startValueRange = 1 // Start values for weights are in range [-startValueRange, startValueRange].

	)

	var (
		inputs, labels []float64
		xys            plotter.XYs
	)
	//https://dev.to/moiz697/how-to-read-and-visualize-csv-data-in-golang-using-gonumplot-5f1h
	file, err := os.Open("data/house_prices.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	reader := csv.NewReader(file)
	if _, err := reader.Read(); err != nil {
		log.Fatal(err)
	}
	data, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	//fmt.Println(data)

	for _, row := range data {
		square, _ := strconv.ParseFloat(row[0], 64)
		price, _ := strconv.ParseFloat(row[2], 64)
		inputs = append(inputs, square)
		labels = append(labels, price)
		xys = append(xys, plotter.XY{X: square, Y: price})
	}
	// f := func(x float64) float64 { return 36*x - 1.75 }
	// for i := 0; i < inputPoints; i++ {
	// 	inputs = append(inputs, inputPointsMinX+(inputPointsMaxX-inputPointsMinX)*rand.Float64())
	// 	labels = append(labels, f(inputs[i])+inputPointsRandY*(1-rand.Float64()*2))
	// 	xys = append(xys, plotter.XY{X: inputs[i], Y: labels[i]})
	// }

	inputsScatter, _ := plotter.NewScatter(xys)

	img := make(chan *image.RGBA, 1) // Have at most one image in the channel.
	render := func(x *image.RGBA) {
		select {
		case <-img: // Drain the channel.
			img <- x // Put the new image in.
		case img <- x: // Or just put the new image in.
		}
	}
	go func() {
		w := startValueRange - rand.Float64()*2*startValueRange
		b := startValueRange - rand.Float64()*2*startValueRange
		var loss plotter.XYs
		for i := 0; i < epochs; i++ {
			y := inference(inputs, w, b)
			loss = append(loss, plotter.XY{
				X: float64(i),
				Y: msl(labels, y),
			})
			lossLines, _ := plotter.NewLine(loss)
			if plotLoss {
				render(Plot(lossLines))
			} else {
				//const extra = (inputPointsMaxX - inputPointsMinX) / 10
				xs := []float64{inputPointsMinX /*- extra*/, inputPointsMaxX /*+ extra*/}
				ys := inference(xs, w, b)
				resLine, _ := plotter.NewLine(plotter.XYs{{X: xs[0], Y: ys[0]}, {X: xs[1], Y: ys[1]}})
				render(Plot(inputsScatter, resLine))
			}
			dw, db := dmsl(inputs, labels, y)
			w += dw * learningRateW
			b += db * learningRateB
			//time.Sleep(30 * time.Millisecond)
			if i%printEveryNthEpochs == 0 {
				fmt.Printf(`Epoch #%d
	loss: %.4f
	dw: %.4f, db: %.4f
	w : %.4f,  b: %.4f
`, i, loss[len(loss)-1].Y, dw, db, w, b)
			}
		}
		fmt.Println(w, b)
	}()

	if err := ebiten.RunGame(&App{Img: img}); err != nil {
		log.Fatal(err)
	}
}

func inference(inputs []float64, w, b float64) (res []float64) {
	for _, x := range inputs {
		res = append(res, w*x+b)
	}
	return res
}

func msl(labels, y []float64) (loss float64) {
	for i := range labels {
		loss += (labels[i] - y[i]) * (labels[i] - y[i])
	}
	return loss / float64(len(labels))
}

func dmsl(inputs, labels, y []float64) (dw, db float64) {
	for i := range labels {
		diff := labels[i] - y[i]
		dw += inputs[i] * diff
		db += diff
	}
	return 2 * dw / float64(len(labels)), 2 * db / float64(len(labels))
}
