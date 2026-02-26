%%
%%  Department of Electrical, Electronic and Computer Engineering
%%  MEng / PhD Research Proposal - Main File
%%  Copyright (C) 2011-2016 University of Pretoria.
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{research_proposal}

%% This class is an extension of the standard article class and will pass
%%   all unknown options to the foundation class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions
\LoadClass[11pt,a4paper]{article}

%% Use the "geometry" package to maximize page usage whilst retaining enough
%%   margin space for staples and editorial notes.
\RequirePackage[a4paper,includeheadfoot,
                top=20mm,left=30mm,right=25mm,bottom=20mm
               ]{geometry}

%% Enable more advanced character protrusion and font expansion, adjustment
%%   of inter-word spacing and additional kerning, etc.
\RequirePackage[tracking=true,kerning=true,spacing=true]{microtype}
\microtypecontext{spacing=nonfrench}

%% Load modern Latin font encoding template T1, as well as additional
%%   symbols.
\RequirePackage{lmodern}
\RequirePackage[T1]{fontenc}
\RequirePackage{textcomp}

%% Use the standard PostScript Type 1 version fonts for Roman, Sans Serif
%%   and Typewriter font classes. This allows the PDF files to be smaller
%%   as the custom TeX fonts do not have to be embedded.
\RequirePackage{mathptmx}
\RequirePackage[scaled=0.92]{helvet}
\RequirePackage{courier}

%% Load AMS mathematics support, fonts and symbols.
\RequirePackage[fleqn]{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amstext}
\RequirePackage{amssymb}
\RequirePackage{bm}

%% Define input language as British English.
\RequirePackage[british]{babel}

%% Load support for advanced control structures in macros.
\RequirePackage{ifthen}

%% Load support for direct PDF generation.
\RequirePackage{ifpdf}

%% Modify section titles and the spacing above/below them, resulting
%%   in significant space-savings. The nobottomtitles command ensures
%%   that no titles appear at the end of a page. The bottomtitlespace
%%   value determines the restricted zone for bottom titles.
\RequirePackage[small,compact,nobottomtitles*]{titlesec}
\renewcommand{\bottomtitlespace}{.12\textheight}
\titleformat*{\section}{\large\bfseries}
\titlelabel{\thetitle.\quad}
\titleformat*{\subsection}{\bfseries}
\titleformat*{\subsubsection}{\bfseries}

%% Enable fancy headers and define a plain style footer with page number
%%   right corner.
\RequirePackage{fancyhdr}
\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[R]{\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

%% Allow importing of EPS figures and enable support for EPS -> PDF
%%   conversion of figures required by pdflatex.
\RequirePackage{epsfig}

%% Pull in support for EPS -> PDF conversion for the UP logo.
\ifpdf
  \usepackage{epstopdf}
\else
\fi

%% Load package to support graphics.
\RequirePackage{graphicx}

%% Enable support for extended list environments.
\RequirePackage{paralist}

% Disable paragraph indent and set default paragraph skip.
\setlength{\parindent}{0 pt}
\setlength{\parskip}{0.8ex} %

% Space below captions
\setlength{\belowcaptionskip}{10 pt}

% Reset maths style parameters.
\setlength{\abovedisplayskip}{4.5pt plus 1pt minus 1pt}
\setlength{\belowdisplayskip}{4.5pt plus 1pt minus 1pt}
\setlength{\abovedisplayshortskip}{0pt plus 1pt}
\setlength{\belowdisplayshortskip}{0pt minus 1pt}

% Set maths equation number to reset per subsection
\numberwithin{equation}{section}

% Include space between rows in equarray
\setlength{\jot}{5 pt}

% Allow extra whitespace in lines, decrease number of hypenations
\sloppy

%% UP specific macros
\newcommand{\up}{University of Pretoria}
\newcommand{\ebit}{Faculty of Engineering, Built Environment and Information Technology}
\newcommand{\eeceup}{University of Pretoria. Department of Electrical, Electronic and Computer Engineering.}
\newcommand{\eece}{Department of Electrical, Electronic and Computer Engineering}

%% Macros for research proposal essential elements.
\newcommand{\rptitle}[1]{\newcommand{\rpthetitle}{#1}}
\newcommand{\rptype}[1]{
  \newcommand{\rpthetype}{#1}
  \ifthenelse{\equal{\rpthetype}{MEng}}{
    \newcommand{\rpthestudyleaderdesignation}{Supervisor}}{}
  \ifthenelse{\equal{\rpthetype}{MSc}}{
    \newcommand{\rpthestudyleaderdesignation}{Supervisor}}{}
  \ifthenelse{\equal{\rpthetype}{PhD}}{
    \newcommand{\rpthestudyleaderdesignation}{Promoter}}{}
}
\newcommand{\rpcandidatename}[1]{\newcommand{\rpthecandidatename}{#1}}
\newcommand{\rpstudentnumber}[1]{\newcommand{\rpthestudentnumber}{#1}}
\newcommand{\rpdate}[1]{\newcommand{\rpthedate}{#1}}
\newcommand{\rpsupervisor}[1]{\newcommand{\rpthesupervisor}{#1}}
\newcommand{\rpeinyear}[1]{\newcommand{\rptheeinyear}{#1}}
\newcommand{\rpeinsupervisor}[1]{\newcommand{\rptheeinsupervisor}{#1}}

%% Standard title / cover page for research proposal.
\newcommand{\maketitlepage}{
{
  \renewcommand{\baselinestretch}{1.2}
  \setlength{\parindent}{0mm}
  \setlength{\parskip}{0mm}
  \centering

  {\bfseries\MakeUppercase{\rpthetitle}}

  \vspace*{\stretch{1}}

  Research proposal: \rpthetype

  \vspace*{\stretch{1}}

  Candidate \\
  \rpthecandidatename

  \vspace*{\stretch{1}}

  Student number \\
  \rpthestudentnumber

  \vspace*{\stretch{1}}

  \eece \\ \up

  \vspace*{\stretch{1}}

  \rpthedate

  \vspace*{\stretch{1}}

  \rpthestudyleaderdesignation \\
  \rpthesupervisor

  \vspace*{\stretch{1}}

  \begin{flushleft}
    \textbf{Declaration regarding intended research and EIN732: Introduction
    to Research} \newline
    \newline
    I completed EIN732 in \rptheeinyear\ under supervision of \rptheeinsupervisor.
    I confirm that my intended research is reflected by the content of the
    attached research proposal.
  \end{flushleft}

  \vspace*{\stretch{2}}

  \begin{tabular}{lp{1cm}ll}

  \makebox[3in]{\hrulefill}             &  & \makebox[1.5in]{\hrulefill} \\
  Candidate (Signature)                 &  & Date \\[2cm]

  \makebox[3in]{\hrulefill}             &  & \makebox[1.5in]{\hrulefill} \\
  \rpthestudyleaderdesignation\ (Signature)  &  & Date \\[2cm]

  \makebox[3in]{\hrulefill}             &  & \makebox[1.5in]{\hrulefill} \\
  Postgraduate Committee (Signature)    &  & Date

  \end{tabular}

  \vspace*{\stretch{4}}
}

\newpage
}

%% Set default styles for document, to single line spacing as required.
\pagestyle{plain}
\pagenumbering{arabic}
\renewcommand{\baselinestretch}{1}

\endinput

%% End of File.
