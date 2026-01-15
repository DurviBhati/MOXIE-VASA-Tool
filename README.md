# LAPSE: Luminex Analyses Personalized Environment

**An AI-Agentic Workflow for Automated, Statistically Rigorous Luminex Data Analysis.**

---

## 🧬 What is LAPSE?
LAPSE is a specialized bioinformatics tool designed to automate the analysis of **Luminex (multiplex immunoassay)** data. Unlike standard tools that simply plot data, LAPSE acts as an intelligent **AI Research Assistant**.

As the development in AI continues, the future in research would be Organization powers AI tool. LAPSE begins with that thought, It uses a **Private AI Agent** (powered by UMich GPT) to guide researchers through complex statistical decisions—such as handling censored data, quality control, and differential expression analysis—ensuring reproducible and statistically valid results.

## 🛑 The Problem
Current workflows for analyzing Luminex cytokine data are flawed:
1.  **Ignored pre-processing:** Data preprocessing is important for correct analyses. Sometimes we often replace values "Below Detection Limit" with `0`, leading to biased results. It requires constant checking of the data to see if it is fit for further analyses.
2.  **Manual Error:** Quality Control (checking bead counts, CV%) is tedious and often skipped.
3.  **Privacy Risks:** Public AI tools cannot be used for sensitive patient data (PHI).
4.  **Static Tools:** Existing software provides rigid outputs without taking in user wants and needs into consideration. Differnt research will involve different analyses type and therefore explaining *why* a result is significant.

## 💡 The LAPSE Solution
LAPSE solves these issues by integrating **Agentic AI** with **Rigorous Biostatistics**:
* **Smart Censoring:** Automatically detects Left-Censored data and suggests **Tobit Regression** or **MLE** instead of simple substitution.
* **Automated QC:** Acts as a guard flagging samples with low bead counts (<35) or high Coefficient of Variation (>20%).
* **Privacy-First:** Built to run with University-hosted AI (e.g., UMich GPT), keeping patient data within the secure firewall.
* **Contextual Analysis:** Uses *Limma-style* linear modeling for differential expression, adjusted for covariates (e.g., "Day -4 vs Day 0"), it asks what the user wants and chooses accordingly.

## 📂 Inputs & Outputs

### **Input**
* **Data File:** Standardized CSV/Excel file containing Luminex raw data.
* **Metadata:** Experimental design info (e.g., Timepoints, Treatment Groups).

### **Output**
An interactive **HTML Analysis Report** containing:
1.  **QC Summary:** Table of flagged/excluded samples.
2.  **Differential Expression:**
    * **Volcano Plots:** Visualizing fold change vs. significance.
    * **Heatmaps:** Clustered cytokine profiles.
    * **Statistical Tables:** Top hits ranked by **FDR-adjusted p-values**.

---
*Developed for the automation of high-throughput immunology data analysis.*

Disclaimer : Used gen AI (mainly gemini) to make the Readme look attractive by adding emojis and pointers, correct the grammatical mistakes.
