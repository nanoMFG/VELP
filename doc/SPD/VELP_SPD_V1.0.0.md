# nanoMFG Software Planning Document
<!-- Replace text below with long title of project:short-name -->
## Machine-learning Assisted Virtual Exfoliation of 2D Materials via Liquid-Phase: MA-VELP
### Target Release: 1.0.0 : Oct 15 , 2019

## Development Team
<!-- Complete table for all team members 
 roles: PI, developer, validation
 status: active, inactive
-->
Name | Role | github user | nanohub user | email | status
---|---|---|---|---|---
Narayana Aluru | PI | - | - | aluru@illinois.edu | active
Alireza Moradzadeh | developer | moradza | moradza | moradza2@illinois.edu | active
Darren Adams | developer | dadamsncsa | dadamsncsa | dadams@illinois.edu | active

**nanoMFG Github Team(s):** @nanoMFG/ma-velp-dev
**nanoHUB Group(s):**

## 1. Introduction
<!-- A  concise description of the current iteration of work. -->
We are employing machine learning algorithm on the high-throughput computational data to improve solvent selection for exfoliation via liquid phase. The code is written in python, and uses standard python libraries such numpy, sklearn, ipywidget, tensorflow 1.x. The GUI is avialable through jupyter notebook.

### 1.1 Purpose and Vision Statement
<!--Why are we building this tool? What is the key benefit? How does it relate to existing tools and existing software? How does it fit into the overall objectives for the nanoMFG node? Who will use it?-->
MA-VELP is a tool to combine data obatined from molecular dynamics (MD) simulations of exfoliation via liquid-phase with machine learning algorithm. After data obatined from MD simulation, they are analyzed using various machine learning algorithms to obtain optimal solvent composition, which facilitates the exfoliation process. Considering the nano-scale of phenomena occurring in the exfoliation process, combination of computational physics and machine learning is one of the best methods to optimize the solvent to further improve the exfoliation process, guiding experimentalist in initial steps with reducing numbers of expensive experimental attempts, especially as commonly used solvents in the LPE process are room temperature ionic liquids (RTILs) which have up to several millions of possible solvents.

#### Why is this tool needed?
The tool is designed with objective to facilitate the design of solvent for exfoliation via liquid-phase, it provides a data-driven framework for following objectives:
   1. predict perforamnce of a given solvent through training a specific machine learning algorithm
   2. allows for design of a solvent composition with optimal performance.

#### What are the key Benefits
It provide an easy GUI for experimentalist to guid them, especially in initial steps. Therefore, reducing number of iterations and costs required during experiment design.

##### How does this tool relate to existing tools and existing software?
The dataset grow in size.

##### How does this project fit into the overall objectives for the nanoMFG node? Who will use it?
The tool main focus is to advance the solvent design for exfoliation via liquid-phase. Experimentalist and nano-manufacturing companies. 


### 1.2 References
<!--List any documents or background material that are relevant.  Links are useful. For instance, a link to a wiki or readme page in the project repository, or link to a uploaded file (doc, pdf, ppt, etc.).-->
https://en.wikipedia.org/wiki/Graphene_production_techniques<br/>
https://en.wikipedia.org/wiki/Intercalation_(chemistry)#Exfoliation<br/>
https://pubs.acs.org/doi/abs/10.1021/acsnano.5b02683<br/>
https://drive.google.com/drive/folders/1RCL-sspuuhA0A5rUhRUl6-UN7yplvEnQ?usp=sharing<br/>






## 2 Overview and Major Planned Features
<!--Provide and overview characterising this proposed release.  Describe how users will interact with each proposed feature.-->
The current release to select the types of cations and anions, they want to study. Futhermore, users will have the option to select the machine-learning algorithms based on the accuracy and needs. Optimization algorithm for finding the optimal solvent is also avaialble through the GUI as well as relative performance of a specific solvent. 

### 2.1 Product Background and Strategic Fit
<!--Provide context for the proposed product.  Is this a completely new projects, or next version of an existing project? This can include a description of any contextual research, or the status of any existing prototype application.  If this SPD describes a component, describe its relationship to larger system. Can include diagrams.-->
The tool is first of its kind to the best of our knoweldge. 

### 2.2 Scope and Limitations for Current Release
<!--List the all planned goals/features for this release.  These should be links to issues.  Add a new subsection for each release.  Equally important, document feature you explicity are not doing at this time-->

Current release requires start from a terminal, in future release it will be replaced with jupyter notebook.

##### 2.2.1 Planned Features
GUI enabled.
Machine Learning Algorithm Selection and Options

#### 2.2.2 Release Notes <!-- optional record of each release -->
##### v#.#.#

#### Release Notes v#.#.#

##### Not Done v#.#.#

### 2.3 Scope and Limitations for Subsequent Releases
<!--Short summary of  future envisioned roadmap for subsequent efforts.-->
Moving to jupyter notebook.
Adding multi-objective cost function. 

### 2.3 Operating Environment
<!--Describe the target environment.  Identify components or application that are needed.  Describe technical infrastructure need to support the application.-->
A Python of 3.0 or higher is required.
### 2.4 Design and Implementation Constraints
<!--This could include pre-existing code that needs to be incorporated ,a certain programming language or toolkit and software dependencies.  Describe the origin and rationale for each constraint.-->
Standard modules of python and tensorflow, scikit learn, and tkinter modules are required.
### 2.5 Components
List of critical variables and functions in the tool
1.	Class Kernel_Optimization(): The role of this class is to fit a kernel-based machine learning model on the MD data based on user selection. 
1.1.	function init(): initialize kernel 
1.2.	function kr_fun(): return fitted machine learning model
1.3.	function read_data(): return dataset
1.4.	function constraint(): applies constraint during optimization
1.5.	function minimize(): minimizes the machine learning model using nedler-mead algorithm
2.	Class DNN():The role of this class is to fit an artificial neural network-based machine learning model on the MD data based on user selection. 
2.1.	function init(): initialize kernel 
2.2.	function kr_fun(): return fitted machine learning model
2.3.	function read_data(): return dataset
2.4.	function constraint(): applies constraint during optimization
2.5.	function minimize(): minimizes the machine learning model using nedler-mead algorithm
2.6.	function mlp(): return value of y based on  the fitted ML model
2.7.	function save()  and load(): save the network after training or load a saved network
3.	Class App(): Handles the GUI launching and combining different classes. 



We generate data from various known function which have analytical solutions, we check whether model can predict analytical solution or not for various cases.

Separately test each function in different classes for its performance.
Test class as a whole.
Test combination of various class.
All the test above is performed on the data set with known solution.
Explain the reasoning behind your testing plan
Starting from most elementary functions in each class we test them separately making sure each function works. Then, the whole class is test and function communication. Then, combination of all the classes together investigated. Through this method, we test the integrity and accuracy at the same time, and it will be easier to know if there is any bug or flaw. 

## 3 User Interaction and Design

### 3.1 Classes of Users
<!--Identify classes (types) of users that you anticipate will use the product.  Provide any relevant context about each class that may influence how the product is used: 
The tasks the class of users will perform
Access and privilege level
Features used
Experience level
Type of interaction
Provide links to any user surveys, questionnaires, interviews, feedback or other relevant information.-->
Three classes of users are considered:
1. Academic users (experimentalist and computationalist): The tool is designed to be simple for user in order to make it easy for experimentalist to use it by selecting the right materials. Computationalist also have the option to add further algorithms as well as data into the dataset of MA-VELP. Basic understanding of coding language and python is required to use the tool. Users can either use it as an educational tool for applied machine learning in physics and chemistry or as a tool to guide their research.
2. Industry users: Similar to experimentalist they can use tool simply and guid their manfacturing toward more efficient one. Basic understanding of coding language and python is required to use the tool. The tool can guide the user toward more efficient process during the manufacturing of 2D materials.
3. Educatioanl users: The tool can be used in the context of various courses such as material discovery and chemical process course. It gives users (students) first involvement with machine learnign and optimization process in the context of physical problems. Students without any implementaiton get to use machine learning algorithm without dealing with implementation steps and experiment with improving a design process.

### 3.2 User Requirements
<!-- Provide a list of issue links to document the main set of user requirements to be satisfied by this release.  Use the user requirement template to draft thense issues.  A well written user requirement should be easy to justify (Rational) and should be testable.  List in order of priority as must have, should have or nice to have for each use case. -->


### 3.3 Proposed User Interface
<!--Could include drawn mockups, screenshots of prototypes, comparison to existing software and other descriptions.-->
<img width="800" src="https://github.com/nanoMFG/VELP/blob/moradza-patch-2/doc/templates/Start_page.png"> <br/>
<img width="800" src="https://github.com/nanoMFG/VELP/blob/moradza-patch-2/doc/templates/Slide8.JPG"> <br/>
<img width="800" src="https://github.com/nanoMFG/VELP/blob/moradza-patch-2/doc/templates/Slide9.JPG">

### 3.4 Documentation Plan
<!-- List planned documentation activities -->

### 3.5 User Outreach Plan
<!-- List upcoming activities designed to elicit user feedback and/or engage new users.  Use issues for activities that will be completed this iteration-->

## 4. Data And Quality Attributes

### 4.1 Data Dictionary
<!--Summarize inputs and outputs for the application.-->
* Dataset Input: Composition of solvents and their corresponding potential of mean force for exfoliation
* User Input: Solvents including types of cations and anions.
* Output: Optimal solvent or Potential of mean force for specific composition of solvent.

### 4.2 Usability and Performance
<!--Summarize usability requirements such as easy of adoption for new users (eg example data),  inline documentation, avoiding errors, efficient interaction, etc.  Describe performance expectations  and/or document challenges.  Note you can reference user requirements from above if needed. -->
Check out the 3.3 Images.

### 4.3 Testing, Verification and Validation
<!--Describe What data is necessary to verify the basic functionality of the application.  Provide a testing plan that includes a list of issues for each planned activity.  Describe data sets that are needed to test validation.-->
The user can replace the dataset with an analytical dataset with known analytical results and check if they can reproduce the same result with the tool.
### 4.4 Uncertainty Quantification
<!--Identify and document possible sources of uncertainty. Categorize with standard labels, such as parametric, structural, algorithmic, experimental, interpolation.

Develop a plan for measuring and documenting uncertainty, e.g., using forward propagation or inverse UQ, and showing it in the application, if applicable.-->
The most fundamental test can be done through selecting a specific composition of solvent and removing it from the dataset, and see how well the tool can predict the output for that specific solvent.

