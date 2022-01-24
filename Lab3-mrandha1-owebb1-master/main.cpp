/*
 * Swarthmore College, CS 87
 * Copyright (c) 2020 Swarthmore College Computer Science Department,
 * Swarthmore PA, Professor Tia Newhall
 */

// This runs the visualizer and calls run on our simulator

// NOTE: you really should not have to change the code in here
//  the only change you may want to make is to the QTSafeViewer
//  argument values based on the size of N
//
#include <qtSafeViewer.h>
#include <dataVisCUDA.h>
#include "firesimulator.h"

int main(int argc, char *argv[]) {

  int width, height, iters, res;

  // init to default values
  width = N;
  height = N;
  iters = NUMITERS;  // this may change based on command line parsing
  /* 1. create QTSafeViewer, pass in window dimensions */
  // if N is 512x512, this size viewer works well:
  QTSafeViewer viewer(600, 500, "Fire!");
  // if N is 800 try this size (the smaller size should work too):
  //QTSafeViewer viewer(900, 800, "Fire!");

  /* 2.a create a DataVisCuda object passing in width and height */
  DataVisCUDA* vis = new DataVisCUDA(width, height);

  /* 2.b create a fireSimulatorKernel object (class derived from Animator) */
  // TODO: implement this constructor and parse command line
  //       arguments in the constructor (or a fucntion it calls)
  Animator* kern = new fireSimulatorKernel(width, height, argc, argv);

  /* 2.c connect Animator (firesSimulator) to DataVisCUDA object */
  vis->setAnimator(kern);

  /* 3. set the viewer's DataVis animation */
  viewer.setAnimation(vis);

  /* get the number of iterations to run the animation */
  // TODO: your command line parsing in the constructor may have a new value
  iters = ((fireSimulatorKernel *)kern)->totalIters();

  /* 4. once everything is all set up, call run on the viewer to
   * start the animation */
  res = viewer.run(iters); // run for iters iterations

  /* 5. Clean-up */
  /* the viewer will clean up the vis object, but explictly delete the kern */
  delete kern;
  return res;
}
