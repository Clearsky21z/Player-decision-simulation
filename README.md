# Player Decision Making Embedding for Similarity and Simulation

## Group Members
- 

## Introduction

This project focuses on analyzing player decision-making in soccer using embedding techniques for similarity analysis and simulation. The goal is to understand how players make decisions in different game contexts and identify similar decision patterns across players.

**Project Links:**
- Notion: https://www.notion.so/Teamspace-Home-26fe0df648af81008817ced88fd6340e
- Motivating ideas: https://www.centralwinger.com/p/substituting-similarity

## Data Source

All data is sourced from **StatsBomb Open Data** (https://github.com/statsbomb/open-data). The dataset includes:
- Match events (passes, shots, carries, etc.)
- 360 freeze-frame data (player positions at the moment of events)
- Match metadata and lineups
- Competition and season information

## Project Structure

### `code/`
Contains the main Python modules for data processing and analysis

### `data/`
Contains the StatsBomb open data:
- **`open-data/`**: Raw JSON files from StatsBomb including events, matches, lineups, and 360 freeze-frame data
- Organized by competition and season IDs

### `images/`
Storage for visualization outputs, plots, and generated images from analysis.

### `notebooks/`
Jupyter notebooks for exploratory data analysis (EDA) and experimentation

### `/sandbox/`
**Note:** The `sandbox/` folders (found in both `code/` and `notebooks/`) are designated spaces where all group members can put their preliminary work that is not yet presentable. This includes experimental code, draft analyses, and work-in-progress notebooks.
