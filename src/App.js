import React, {Component} from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

const TOTAL=250;
const HEIGHT = 400;
const WIDTH = 500;
const PIPE_WIDTH = 50;
const MIN_PIPE_HEIGHT = 20;

class Pipe{

   constructor(ctx, height){
      this.ctx = ctx;
      this.isDead = false;

      this.x = WIDTH;
      this.y = height ? HEIGHT - height : 0;
      this.width = PIPE_WIDTH;
      this.height = height || MIN_PIPE_HEIGHT
      + (Math.random() * 240) +30;

   }

   draw(){

      this.ctx.fillStyle = "#000";
      this.ctx.fillRect(this.x, this.y, this.width, this.height);

   }

   update(){
      this.x = this.x - 1;
      if((this.x + PIPE_WIDTH) < 0){
         this.isDead= true;
      }
   }
}

class Bird{

   constructor(ctx, brain){
      this.ctx = ctx;
      this.isDead = false;
      this.score=0;
      this.fitness=0;
      this.x = 150;
      this.y = 150;
      this.gravity = 0;
      this.velocity =0.1;


      if (brain) {
        this.brain = brain.copy();
        // this.mutate();
      } else {
        this.brain = new NeuralNetwork(5, 8, 2);
      }
   }

   dispose() {
    this.brain.dispose();
  }

  mutate() {
    this.brain.mutate(0.1);
  }

  think(pipes){
    let closest = null;
    let closestD = Infinity;
    for (let i = 0; i < pipes.length; i++) {
      let d = pipes[i].x + pipes[i].width - this.x;
      if (d < closestD && d > 0) {
        closest = pipes[i];
        closestD = d;
      }
  }

  let inputs = [
    this.y / HEIGHT,
    closest.height / HEIGHT,
    closest.width / HEIGHT,
    closest.x / WIDTH,
    this.velocity / 10
  ];
  // console.log(inputs);
   let output = this.brain.predict(inputs);
  // console.log(output);

  if (output[0] > output[1]) {
     this.jump();
   }
}

   draw(){
      this.ctx.fillStyle= 'red';
      this.ctx.beginPath();
      this.ctx.arc(this.x, this.y, 7, 0 ,2 * Math.PI);
      this.ctx.fill();
   }

   update(){
      this.score += 1;
      this.gravity += this.velocity;
      this.y += this.gravity;

      if (this.y < 0){
        this.y=0;
      }else if (this.y > HEIGHT) {
        this.y= HEIGHT;
      }
      // this.think();
   }

   jump = () => {
      this.gravity= -3.5;
   }
}

class NeuralNetwork{
    constructor(a,b,c,d){
      if (a instanceof tf.Sequential){
        this.model = a;
        this.input_nodes = b;
        this.hidden_nodes = c;
        this.output_nodes = d;
      } else {
      this.input_nodes = a;
      this.hidden_nodes = b;
      this.output_nodes = c;
      this.model = this.createModel();
      }
    }

    copy(){
      tf.tidy(()=>{
        const modelCopy =  this.createModel();
        const weights = this.model.getWeights();
        const weightCopies=[];
        for (var i = 0; i < weightCopies.length; i++) {
          weightCopies[i]= weights[i].clone();
        }
        modelCopy.setWeights(weights);
        return new NeuralNetwork(modelCopy, this.input_nodes, this.hidden_nodes, this.output_nodes);
      });
    }

    mutate(rate) {
      tf.tidy(() => {
        const weights = this.model.getWeights();
        const mutatedWeights = [];
        for (let i = 0; i < weights.length; i++) {
          let tensor = weights[i];
          let shape = weights[i].shape;
          let values = tensor.dataSync().slice();
          for (let j = 0; j < values.length; j++) {
            if (Math.random(1) < rate) {
              let w = values[j];
              values[j] = w + Math.randomGaussian();
            }
          }
          let newTensor = tf.tensor(values, shape);
          mutatedWeights[i] = newTensor;
        }
        this.model.setWeights(mutatedWeights);
    });
  }

  dispose(){
    this.model.dispose();
  }

    predict(inputs){
      return tf.tidy(()=>{
        const xs = tf.tensor2d([inputs]);
        const ys = this.model.predict(xs);
        const outputs = ys.dataSync();
        return outputs;
      });
    }

    createModel(){
      const model = tf.sequential();
      const hidden = tf.layers.dense({
        units: this.hidden_nodes,
        inputShape: [this.input_nodes],
        activation: 'sigmoid'
      });
      model.add(hidden);
      const output = tf.layers.dense({
        units: this.output_nodes,
        activation: 'softmax'
      });
      model.add(output);
      return model;
      // this.model.compile({});
    }
}

class App extends Component{

   constructor(props) {
      super(props);

      this.draw = this.draw.bind(this);
      this.update = this.update.bind(this);
      this.game = this.game.bind(this);
      this.createPipes = this.createPipes.bind(this);
      this.birdState = this.birdState.bind(this);
      this.createBirds = this.createBirds.bind(this);

      this.myCanvas = React.createRef();
      this.frame=0;
      this.pipes =[];
      this.birds = [];
      this.deadBirds=[];
   }

   componentDidMount(){
      tf.setBackend('cpu');
      this.startGame();
   }

   startGame = () => {

     clearInterval(this.loop);
     const ctx = this.myCanvas.current.getContext("2d");
     ctx.clearRect(0,0,WIDTH,HEIGHT);

     this.pipes = this.createPipes();
     this.birds = this.createBirds();
     this.loop = setInterval(this.game, 1000/60);

   }

   // onKeyDown = (e) => {
   //    if(e.code === 'Space'){
   //       this.birds[0].jump();
   //    }
   // }

   createPipes(){
     var ctx = this.myCanvas.current.getContext("2d");
     const firstpipe = new Pipe(ctx, null);
     const secondPipeHeight = HEIGHT - firstpipe.height - 80;
     const secondpipe = new Pipe(ctx, secondPipeHeight);
     return [firstpipe, secondpipe]
   }

   createBirds(){
     const birds = [];
     var ctx = this.myCanvas.current.getContext("2d");
      for (let i = 0; i < 200; i += 1) {
        const brain = this.deadBirds.length && this.pickOne().brain;
        const newBird = new Bird(ctx, brain);
        birds.push(newBird);
      }
      return birds;
   }

   game(){

      this.draw();
      this.update();

   }

   birdState(){
      this.birds.forEach((bird) => {
         this.pipes.forEach((pipe) => {

            if( bird.y <= 0 || bird.y >= HEIGHT || (bird.x >= pipe.x && bird.x <= pipe.x + pipe.width &&
               bird.y >= pipe.y && bird.y <= pipe.y + pipe.height)){

               bird.isDead= true;
            }
         });
      });
   }

   draw(){
      var ctx = this.myCanvas.current.getContext("2d");
      ctx.clearRect(0,0,WIDTH,HEIGHT);

      this.pipes.forEach(pipe => pipe.draw());
      this.birds.forEach(bird => bird.draw());
   }

   update(){
      this.frame++;
      if(this.frame % 200 === 0){
         const pipes =this.createPipes();
         this.pipes.push(...pipes);
      }


      this.pipes.forEach(pipe => pipe.update());
      this.birds.forEach(bird => bird.update());
      this.birds.forEach(bird => bird.think(this.pipes));


      this.pipes = this.pipes.filter(pipe => !pipe.isDead);

      this.birdState();
      this.deadBirds.push(...this.birds.filter(bird => bird.isDead));
      this.birds = this.birds.filter(bird => !bird.isDead);

      if (this.birds.length === 0){
        let total =0;

        this.deadBirds.forEach(deadBird => {total += deadBird.score});

        this.deadBirds.forEach(deadBird => {deadBird.fitness = deadBird.score / total});
        this.startGame();
      }
   }

   pickOne = () => {
    let index = 0;
    let r = Math.random();
      while (r > 0) {
        r -= this.deadBirds[index].fitness;
        index += 1;
      }
      index -= 1;
    return this.deadBirds[index];
  }


render(){
  return (
    <div className="App">

      <canvas
      ref={this.myCanvas}
      width={WIDTH} height={HEIGHT}
      style={{marginTop: '20px', border:'1px solid #c3c3c3'}}>
      </canvas>

      <div onClick={() => this.setState({})}>
         {this.frame}
      </div>

    </div>
  );
}
}

export default App;
