function neuralNetwork() {

    // An array of n zeros
    function zeros(n) {
        var ret = [];
        while(ret.length < n) ret.push(0);
        return ret;
    }

    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    function sigmoidDerivative(x) {
        return x * (1 - x);
    }

    const activation = sigmoid;
    const activationDerivative = sigmoidDerivative;

    // Create a randomly initialized neuron with n synapse (inputs)
    function createNeuron(nInputs) {
        return zeros(nInputs).map(() => Math.random() * 2.0 - 1.0);
    }

    function neuronWeight(neuron, index) {
        return neuron[index];
    }

    function neuronForward(neuron, inputs) {
        return activation(inputs.reduce((sum, input, index) => {
            return sum + input * neuronWeight(neuron, index);
        }, 0));
    }

    function layerForward(layer, inputs) {
        return layer.map((neuron) => neuronForward(neuron, inputs));
    }

    function layerForwardArray(layer, inputsArray) {
        return inputsArray.map(layerForward.bind(null, layer));
    }

    function neuronForwardArray(neuron, inputsArray) {
        return inputsArray.map(neuronForward.bind(null, neuron));
    }

    function createLayer(nInputs, nNeurons) {
        return zeros(nNeurons).map(() => createNeuron(nInputs));
    }

    function layerNeuron(layer, index) {
        return layer[index];
    }

    function create(layerSizes) {
        /*
        if (nNeurons > 1)
            return [
                createLayer(nInputs, nNeurons),
                createLayer(nNeurons, 1)
            ];
        else
            return [ createLayer(nInputs, 1) ]
        */
        return layerSizes.slice(1).map((nNeurons, index) =>
            createLayer(layerSizes[index], nNeurons)
        );
    }

    function train(nIteration, network, trainingInputsArray, trainingOutputArray) {
        while (nIteration-- > 0)
            network = trainIteration(network, trainingInputsArray, trainingOutputArray);
        return network;
    }

    function forward(network, inputs) {
        return network.reduce(
            (x, layer) => layerForward(layer, x),
            inputs);
    }

    function forwardArray(network, inputsArray) {
        return inputsArray.map(forward.bind(null, network));
    }

    function test(name, network, testInputsArray, testOutputsArray) {
        const actual = applyAll(Math.round, forwardArray(network, testInputsArray));
        const expect = testOutputsArray;
        const stats = flatten(sub(actual, expect)).reduce((stats, value) => {
            const ok = (value == 0);
            const success = stats.success + ok;
            const failure = stats.failure + 1 - ok;
            const total = stats.total + 1;
            return { success, failure, total };
        }, { success:0, failure:0, total:0 });

        console.log(name + ": Success=" + Math.round(1000 * stats.success / stats.total) / 10 + "%");
    }

    function dot(m1 /* h1 x w1 */, m2 /* h2 x w2 */) {
        if (m1[0].length != m2.length) {
            throw new Error("dot impossible: w1 != h2");
        }
        // for each row,
        return m1.map((row1, rowIndex) => {
            // produce a new row of size w2
            return m2[0].map((dummy, colIndex) => {
                return m2.reduce((sum, row2, rowIndex2) => {
                    return sum + row1[rowIndex2] * row2[colIndex];
                }, 0);
            });
        });
    }

    function mul(coef, m) {
        if (typeof coef == 'number')
            return m.map((row) => row.map((cell) => cell * coef));
        else {
            if (m.width != coef.width || m[0].width != coef[0].width)
                throw new Error("cant mul matrices of different sizes");
            return m.map((row, rowIndex) => row.map((cell, colIndex) => cell * coef[rowIndex][colIndex]));
        }
    }

    function flatten(m) {
        return m.reduce((array, row) => array.concat(row), []);
    }

    function sub(m1, m2) {
        if (m1.width != m2.width || m1[0].width != m2[0].width)
            throw new Error("cant substract matrices of different sizes");
        return m1.map((row, rowIndex) => row.map((cell, colIndex) => cell - m2[rowIndex][colIndex]));
    }

    function add(m1, m2) {
        if (m1.width != m2.width || m1[0].width != m2[0].width)
            throw new Error("cant add matrices of different sizes");
        return m1.map((row, rowIndex) => row.map((cell, colIndex) => cell + m2[rowIndex][colIndex]));
    }

    function applyAll(fn, m) {
        return m.map((row) => row.map((cell) => fn(cell)));
    }

    function transpose(m) {
        return m[0].map((dummy, colIndex) => m.reduce((out, row, rowIndex) => out.concat(row[colIndex]), []));
    }

    function debugMatrix(matrix, name, fullDisplay) {
        if (typeof matrix[0][0] != 'number') {
            console.log(name + " IS NOT A MATRIX!");
            console.log(matrix);
            return;
        }
        console.log(name + " (" + matrix.length + "x" + matrix[0].length + ")");
        const hasNaN = matrix.reduce((hasNaN, row) => {
            return row.reduce((hasNaN, cell) => {
                return hasNaN || isNaN(cell) || (cell == undefined);
            }, hasNaN);
        }, false);
        if (hasNaN) {
            console.log(name + " IS BUGGY!");
            fullDisplay = true;
        }
        if (fullDisplay)
            console.log(matrix);
    }

    function trainIteration(network, trainingInputs, trainingOutputs) {

        let x = trainingInputs;
        const outputs = network.map((layer) => x = layerForwardArray(layer, x));
        const inputs = [ trainingInputs ].concat(outputs.slice(0, outputs.length - 1));

        const error = [];
        const delta = [];
        const weightOffsets = [];

        for (let i = network.length - 1; i >= 0; --i) {
            error[i] = (i == network.length - 1)
                ? sub(trainingOutputs, outputs[i])
                : dot(delta[i + 1], network[i + 1]);
            delta[i] = mul(error[i], applyAll(activationDerivative, outputs[i]));
            weightOffsets[i] = transpose(dot(transpose(inputs[i]), delta[i]));
        }

        //debugMatrix(trainingInputs, "trainingInputs");
        //debugMatrix(trainingOutputs, "trainingOutputs", true);
        //debugMatrix(network[0], "network[0]");
        //debugMatrix(network[1], "network[1]");
        //debugMatrix(inputs[0],  "inputs[0]");
        //debugMatrix(inputs[1],  "inputs[1]");
        //debugMatrix(outputs[0], "outputs[0]");
        //debugMatrix(outputs[1], "outputs[1]", true);
        //debugMatrix(error[1], "error[1]", true);
        //debugMatrix(delta[1], "delta[1]");
        //debugMatrix(error[0], "error[0]");
        //debugMatrix(delta[0], "delta[0]");
        //debugMatrix(weightOffsets[1], "weightOffsets[1]");
        //debugMatrix(weightOffsets[0], "weightOffsets[0]");

        return network.map((layer, index) => add(layer, weightOffsets[index]));
    }

    return {
        create, test, train, forward, forwardArray
    };
}

const { create, test, train, forward } = neuralNetwork();

function problem1() {
    // most basic: copy over value from first column
    const testInputsArray  = [[1,0,0],[0,1,1],[1,1,1]];
    const testOutputsArray = [ [1], [0], [1] ];

    const trainingInputsArray = [[ 1, 0, 0 ], [ 0, 1, 1 ], [ 1, 1, 1], [ 0, 0, 1], [ 1, 1, 0 ], [ 0, 1, 0 ]];
    const trainingOutputArray = [ [1], [0], [1], [0], [1], [0] ];

    const network = create([3, 3, 1]);
    const trainedNetwork = train(500, network, trainingInputsArray, trainingOutputArray);

    test("problem1", trainedNetwork, testInputsArray, testOutputsArray);
}

function problem2() {
    // 1bit sum of the 3 columns
    const testInputsArray = [[1,0,0],[0,1,1],[1,1,1],[1,1,0]];
    const testOutputsArray = [ [1,0], [0,1], [1,1], [0,1] ];

    const trainingInputsArray = [[ 1, 0, 0 ], [ 0, 1, 1 ], [ 1, 1, 1], [ 0, 0, 1], [ 1, 1, 0 ], [ 0, 1, 0 ]];
    const trainingOutputArray = [ [1,0], [0,1], [1,1], [1,0], [0,1], [1,1] ];

    const network = create([3, 6, 2]);
    const trainedNetwork = train(500, network, trainingInputsArray, trainingOutputArray);

    test("problem2", trainedNetwork, testInputsArray, testOutputsArray);
}

function problem3() {

    function one(a,b) { return a == b ? 1 : 0; }
    function bit(v) {
        return [
            one(v, 0),
            one(v, 1),
            one(v, 2),
            one(v, 3),
            one(v, 4),
            one(v, 5),
            one(v, 6),
            one(v, 7),
            one(v, 8),
            one(v, 9),
            one(v, 10),
            one(v, 11),
            one(v, 12),
            one(v, 13),
            one(v, 14),
            one(v, 15)
        ];
        // return [ v & 1, (v >> 1) & 1, (v >> 2) & 1, (v >> 3) & 1, (v >> 4) & 1, (v >> 5) & 1, (v >> 6) & 1, (v >> 7) & 1 ];
    }
    function fromBit(v) {
        v = v.map(Math.round);
        // return v[0] + 2*v[1] + 4*v[2] + 8*v[3] + 16*v[4] + 32*v[5] + 64*v[6] + 128*v[7];
        return v[0] ? 0 : v[1] ? 1 : v[2] ? 2 : v[3] ? 3 : v[4] ? 4 : v[5] ? 5 : v[6] ? 6 : v[7] ? 7 :
            v[8] ? 8 : v[9] ? 9 : v[10] ? 10 : v[11] ? 11 : v[12] ? 12 : v[13] ? 13 : v[14] ? 14 : v[15] ? 15 : -1;
    }

    const trainingInputsArray = [];
    const trainingOutputArray = [];
    function add(i,o) {
        trainingInputsArray.push(i2b(i));
        trainingOutputArray.push(o2b(o));
    }
    function i2b(i) { return bit(Math.round(i)); }
    function o2b(o) { return bit(Math.round(o)); }
    function b2o(b) { return fromBit(b); }
    /* function o2b(o) { return bit(Math.round(8 * Math.log(1 + o))); }
    function b2o(b) { return Math.round(Math.exp(fromBit(b) / 8) - 1); } */

    add(8, 8.5);
    add(3, 1.5);
    add(2, 2);
    add(8, 3);
    add(5, 2.5);
    add(2, 0.4);
    add(1, 0.5);
    add(1, 1);
    add(2, 0.6);
    add(1, 5);
    add(1, 1.75);
    add(2, 1.3);
    add(2, 0.5);
    add(1, 1);
    add(3, 2.5);
    add(1, 0.8);
    add(2, 0.7);

    const network = create([16, 16]);
    const trainedNetwork = train(1000, network, trainingInputsArray, trainingOutputArray);
    for (let i = 0; i <= 15; ++i) {
        //console.log(i, o2b(i), b2o(o2b(i)));
        console.log(i, b2o(forward(trainedNetwork, i2b(i))));
    }
}

problem1();
problem2();
problem3();
