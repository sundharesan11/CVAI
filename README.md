### CV Matcher AI

** Introduction: **

- Create an AI agent to find top resumes matching a given job description.


** Detail: **

- Given a list of resumes and a job description, the agent should return top N resumes based on
how well they match the job in descending order of match score.


** Input: **

1. List of resumes (~100) - PDF files
2. A job description - PDF


** Output: **

- Resumes sorted by closeness/proximity score to the job description.


** Model Fine Tuning: **

- Provide a way to fine tune the model based on human feedback. 
- Humans will mark the resumes with a rating (BEST, GOOD, AVERAGE, BAD) on how well they match the job description.
- Given these human ratings, the model should fine tune itself.
