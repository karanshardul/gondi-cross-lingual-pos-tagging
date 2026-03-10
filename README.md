# Cross-Lingual POS Tagging for Gondi using Multilingual Embedding Projection

This project implements a cross-lingual Part-of-Speech (POS) tagging system for the low-resource Gondi language using multilingual contextual embeddings from mBERT.

Since annotated resources for Gondi are limited, the system transfers syntactic knowledge from four high-resource languages: Hindi, Telugu, Tamil, and Marathi.

The method projects Gondi tokens into the multilingual embedding space of mBERT and predicts POS tags using k-nearest neighbor similarity with POS-labeled embeddings.
