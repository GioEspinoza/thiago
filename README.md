# Thiago: Intelligent Child Tracking and Assistance System

## Overview

Thiago is an AI-powered child monitoring and assistance system designed to identify, track, and monitor a specific child in real time using computer vision and machine learning.

The project's long-term goal is to create a mobile robotic platform capable of following a designated child, analyzing behavior, providing interaction, and notifying caregivers when necessary.

This repository currently contains the software perception pipeline responsible for human detection, tracking, and identity recognition.

---

## Features

### Human Detection

Uses YOLOv8 to perform real-time human detection from a live camera feed.

* Detects people in each frame
* Filters detections to the "person" class
* Draws bounding boxes around detected individuals

### Multi-Object Tracking

Uses BoTSORT to maintain tracking IDs across video frames.

* Assigns temporary IDs to detected individuals
* Maintains identity consistency between frames
* Supports tracking multiple people simultaneously

### Target Selection

Implements custom target selection logic.

Current implementation:

* Detect all people in frame
* Determine the tallest detected individual
* Designate selected individual as the target

Future versions will support manual and automatic target acquisition.

### Person Re-Identification (ReID)

Uses TorchReID with OSNet to generate appearance embeddings for detected individuals.

* Extracts a 512-dimensional feature vector for each detected person
* Creates appearance-based identity representations
* Enables identity matching across tracking failures

### Identity Recognition

Planned implementation:

* Generate reference embeddings from known images
* Store identity embeddings locally
* Compare live embeddings against known identities
* Identify and track specific individuals

Example:

```text
Detected Person
        ↓
Generate Embedding
        ↓
Compare Against Stored Identity Embeddings
        ↓
Match Found?
        ↓
YES → Identify as Target Child
NO  → Unknown Individual
```

---

## Software Architecture

```text
Camera Feed (OpenCV)
          ↓
YOLOv8 Human Detection
          ↓
BoTSORT Tracking
          ↓
Person Cropping
          ↓
OSNet Feature Extraction
          ↓
Identity Comparison
          ↓
Target Selection
          ↓
Visualization & Tracking Output
```

---

## Current Technology Stack

### Computer Vision

* OpenCV
* YOLOv8
* BoTSORT

### Machine Learning

* PyTorch
* TorchReID
* OSNet

### Development

* Python
* Git
* GitHub
* Linux

---

## Project Roadmap

### Phase 1: Human Detection

* [x] Camera integration
* [x] YOLOv8 detection
* [x] Bounding box visualization

### Phase 2: Human Tracking

* [x] BoTSORT integration
* [x] Tracking IDs
* [x] Target selection logic

### Phase 3: Identity Recognition

* [ ] Reference image dataset
* [ ] Identity embedding generation
* [ ] Embedding storage
* [ ] Similarity matching
* [ ] Persistent identity recognition

### Phase 4: Behavioral Analysis

* [ ] Activity monitoring
* [ ] Behavioral event detection
* [ ] Caregiver notifications

### Phase 5: Robotic Integration

* [ ] Arduino integration
* [ ] Motor control
* [ ] Autonomous following
* [ ] Speaker system
* [ ] Sensor integration

---

## Motivation

Thiago was created to explore how modern computer vision, machine learning, and robotics can be combined to build intelligent systems capable of assisting children with special needs.

The project serves as both a learning platform and a foundation for future assistive technologies.

---

## Authors

### Software Development

Giovanni Espinoza

### Hardware Development

Giselle Espinoza
