# CFR-dependent items (set aside from the benchmark)

Per issue [#7](https://github.com/bwaldon/eabench/issues/7): the benchmark should be
answerable from the FERPA statute (20 U.S.C. § 1232g) alone. The three items below
were found to genuinely **require** a Code of Federal Regulations (34 CFR Part 99)
provision to be answerable — the decisive rule has no statutory equivalent — so they
are removed from `benchmark.jsonl` and preserved here rather than deleted. Their
former slots (task_ids 41, 43, 50) are reused by three new FERPA-statute-only items.

If a future benchmark variant admits regulation sources, these can be reinstated as-is.

| former task_id | topic | why CFR is load-bearing |
|---|---|---|
| 41 | grade-amendment hearing procedure | the decisive procedure — hearing under **§ 99.22**, written statement under **§ 99.21(b)(2)** — is regulation-only; the statute (§ 1232g(a)(2)) grants the right but not the procedure |
| 43 | records of a deceased eligible student | the only *codified* authority is **§ 99.5(a)(1)**; the death-lapse rule itself rests on FPCO/PTAC guidance, not statutory text |
| 50 | dual-enrollment / dependent-student access | turns on the **§ 99.5(b)** dual-enrollment carve-out and the **§ 99.31(a)(8)** dependent-student condition, neither of which is in the statute |

---

## Former item 41 — grade-amendment hearing

Query:

"A parent has formally requested under FERPA that we change their child's grade in algebra from a C to a B. They believe the teacher graded unfairly. Do we have to make the change?"

Statutory provision

20 U.S.C. § 1232g(a)(2) No funds shall be made available under any applicable program to any educational agency or institution unless the parents of students … are provided an opportunity for a hearing by such agency or institution … to challenge the content of such student's education records, in order to insure that the records are not inaccurate, misleading, or otherwise in violation of the privacy rights of students, and to provide an opportunity for the correction or deletion of any such inaccurate, misleading, or otherwise inappropriate data contained therein and to insert into such records a written explanation of the parents respecting the content of such records.

34 CFR § 99.21(a) An educational agency or institution shall, on request, provide an opportunity for a hearing under § 99.22 to challenge the content of a student's education records on the grounds that the information contained in the education records is inaccurate, misleading, or in violation of the privacy rights of the student.

Expected follow-up question:

Does the parent's challenge concern (a) a clerical or factual error in the recording of the grade (the grade was misentered, or the grade book reflects work that was not the student's), or (b) the teacher's substantive academic judgment of the quality of the work?

FERPA's amendment right is limited to records that are "inaccurate, misleading, or in violation of the privacy rights of the student." The Department's guidance and PTAC have consistently held that the right to seek amendment does not extend to substantive academic judgments. A challenge to a teacher's evaluation of work quality is outside FERPA's scope.

Query Continuation

"The parent's letter does not allege that the grade was misentered or that anything was clerically wrong. The parent's argument is that the teacher's grading rubric was applied too harshly and that the student's work merited a higher grade."

Expected Final Answer

The school is not required to amend the grade. FERPA's amendment right under §1232g(a)(2) and 34 CFR §§ 99.20–99.22 reaches only records that are inaccurate, misleading, or in violation of the student's privacy rights. A challenge to the substantive academic judgment underlying a grade — that is, the teacher's evaluation of the quality of the work — is not within that scope. The school must, however, provide the parent with a hearing under § 99.22 if requested, even though the hearing is unlikely to result in amendment, and must allow the parent to insert a written statement of the parent's view into the record under § 99.21(b)(2). If the parent's challenge had been clerical (e.g., that the grade entered did not match the grade the teacher actually assigned, or that work attributed to the student was actually another student's), FERPA's amendment process would be the right mechanism and the school would have to consider the merits.

Concepts in explanation

- SCOPE LIMITER: FERPA's amendment right does not extend to substantive academic judgments.
- PROCEDURAL OBLIGATION: Even when the school will not amend, it must offer a hearing and allow a written statement from the parent.

---

## Former item 43 — deceased eligible student

Query:

"A documentary filmmaker has asked our college for the academic records of a former student who died last year at age 24. They say their work concerns the student's posthumous reputation. Are we permitted to share?"

Statutory provision

FPCO/PTAC Guidance (informed by 20 U.S.C. § 1232g(d) and common law principles): Once an eligible student dies, FERPA rights lapse. FERPA does not protect the education records of a deceased eligible student, and the institution may disclose such records at its discretion or consistent with State law.

34 CFR § 99.5(a)(1) When a student becomes an eligible student, the rights accorded to, and consent required of, parents under this part transfer from the parents to the student.

Expected follow-up question:

Was the student an "eligible student" under FERPA at the time of death — that is, age 18 or older, or attending a postsecondary institution at any age — or did the student die while a parent still held FERPA rights (under 18 and not in postsecondary)?

The Department's interpretation is that FERPA rights of an eligible student lapse on death; FERPA no longer protects the records, and the institution has discretion to disclose subject to State law. If the student died before becoming an eligible student, the analysis is more complicated and the parent's FERPA rights may continue or transfer at common law; State law and institutional policy then govern.

Query Continuation

"The student was 24 at the time of death and had been an eligible student for years. State law has no specific provision governing post-mortem disclosure of education records."

Expected Final Answer

FERPA does not protect the records of this deceased eligible student. The institution may disclose the records at its discretion, subject to its own institutional policy and any applicable State law (here, none). Most institutions adopt a written policy governing post-mortem disclosure — typically requiring approval at a senior level, defining categories of permitted recipients (next of kin, executors, researchers under specified conditions), and considering the student's known wishes if any. The institution should consult its own policy before disclosing to a documentary filmmaker; FERPA is no longer an obstacle, but policy and reputational considerations remain. If the student had been a minor at death (and not enrolled in a postsecondary institution), the analysis would differ and parental consent considerations would persist.

---

## Former item 50 — dual-enrollment / dependent-student access

Query:

"I'm a registrar at a community college. We have a 17-year-old student who is dual-enrolled at our institution and at the local high school. The student's parent has asked to see the student's grades from our courses. Do they have a right to those records under FERPA?"

Statutory provision

20 U.S.C. § 1232g(d) For the purposes of this section, whenever a student has attained eighteen years of age, or is attending an institution of postsecondary education, the permission or consent required of and the rights accorded to the parents of the student shall thereafter only be required of and accorded to the student.

34 CFR § 99.5(b) The Act and this part do not prevent educational agencies or institutions from disclosing education records, or personally identifiable information from education records, to a parent without the prior written consent of an eligible student if the disclosure meets the conditions in § 99.31(a)(8) [parents of dependent students] or any of the other conditions in § 99.31.

Expected follow-up question:

(1) At which institution is the parent seeking to view the student's records — the high school, the community college, or both? (2) Is the student claimed as a dependent on the parent's most recent federal tax return?

A student attending a postsecondary institution is an eligible student under FERPA, regardless of age — so rights as to the postsecondary records have already transferred to the student. The parent does not retain access by virtue of the student's age (under 18) at the postsecondary level. The parent does retain rights as to the high-school records in the dual-enrollment context. The (b)(1)(H) dependent-student exception is the only common path for parental access to postsecondary records without consent.

Query Continuation

"The parent is asking for the community college grades. The student is claimed as a dependent on the parent's most recent federal tax return."

Expected Final Answer

The community college may disclose the student's records to the parent under §1232g(b)(1)(H) and 34 CFR § 99.31(a)(8), the dependent-student exception. Although the student is an eligible student at the postsecondary institution (and therefore the parent does not have inherent rights under §1232g(a) at that institution), the dependent-student exception is permissive and applies because the student is claimed as a dependent on the parent's federal tax return. The college may verify dependent status by asking the parent for relevant tax-return documentation. Disclosure under this exception is permissive, not required, and the college's institutional policy may add further conditions (e.g., notice to the student before disclosure). The high school, by contrast, must allow the parent to inspect/review the student's high-school records as a matter of course (the student is under 18 and not a postsecondary-only student for high-school FERPA purposes — see 34 CFR § 99.5(b) which also expressly permits the high school and the postsecondary institution to exchange information about the dual-enrolled student). If the student were not a dependent for tax purposes, the college could not disclose to the parent without the student's written consent.

Concepts in explanation

- RIGHTS-TRANSFER RULE: Postsecondary attendance triggers transfer of rights regardless of age.
- PERMISSIVE EXCEPTION: The dependent-student exception is permissive, not mandatory.
- DUAL-ENROLLMENT CARVE-OUT: § 99.5(b) preserves information exchange between high school and postsecondary institutions for dual-enrolled students.
