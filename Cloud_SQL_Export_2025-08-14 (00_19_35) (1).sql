--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: chapters; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.chapters (
    chapter_id integer NOT NULL,
    vector_id character varying(255)
);


--
-- Name: chapters_chapter_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.chapters_chapter_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: chapters_chapter_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.chapters_chapter_id_seq OWNED BY public.chapters.chapter_id;


--
-- Name: classes; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.classes (
    class_id integer NOT NULL,
    name character varying(100) NOT NULL,
    number_of_students integer DEFAULT 0
);


--
-- Name: classes_class_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.classes_class_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: classes_class_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.classes_class_id_seq OWNED BY public.classes.class_id;


--
-- Name: questions; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.questions (
    question_id integer NOT NULL,
    chapter_id integer,
    topic_id integer,
    question text
);


--
-- Name: questions_question_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.questions_question_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: questions_question_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.questions_question_id_seq OWNED BY public.questions.question_id;


--
-- Name: students; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.students (
    student_id integer NOT NULL,
    name character varying(100) NOT NULL,
    photo_id character varying(255),
    class_id integer,
    attendance integer
);


--
-- Name: students_student_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.students_student_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: students_student_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.students_student_id_seq OWNED BY public.students.student_id;


--
-- Name: subjects; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.subjects (
    subject_id integer NOT NULL,
    name character varying(100) NOT NULL,
    class_id integer,
    number_of_topics integer DEFAULT 0
);


--
-- Name: subjects_subject_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.subjects_subject_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: subjects_subject_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.subjects_subject_id_seq OWNED BY public.subjects.subject_id;


--
-- Name: topics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.topics (
    topic_id integer NOT NULL,
    name character varying(255) NOT NULL,
    subject_id integer,
    chapter_id integer
);


--
-- Name: topics_topic_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.topics_topic_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: topics_topic_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.topics_topic_id_seq OWNED BY public.topics.topic_id;


--
-- Name: weak_topics; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.weak_topics (
    topic_id integer NOT NULL,
    student_id integer NOT NULL,
    weakness integer,
    CONSTRAINT weak_topics_weakness_check CHECK ((weakness = ANY (ARRAY[0, 1])))
);


--
-- Name: chapters chapter_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chapters ALTER COLUMN chapter_id SET DEFAULT nextval('public.chapters_chapter_id_seq'::regclass);


--
-- Name: classes class_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classes ALTER COLUMN class_id SET DEFAULT nextval('public.classes_class_id_seq'::regclass);


--
-- Name: questions question_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions ALTER COLUMN question_id SET DEFAULT nextval('public.questions_question_id_seq'::regclass);


--
-- Name: students student_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.students ALTER COLUMN student_id SET DEFAULT nextval('public.students_student_id_seq'::regclass);


--
-- Name: subjects subject_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.subjects ALTER COLUMN subject_id SET DEFAULT nextval('public.subjects_subject_id_seq'::regclass);


--
-- Name: topics topic_id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics ALTER COLUMN topic_id SET DEFAULT nextval('public.topics_topic_id_seq'::regclass);


--
-- Data for Name: chapters; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.chapters (chapter_id, vector_id) FROM stdin;
1	chapter1_math_grade4
2	chapter2_math_grade4
3	chapter3_math_grade4
\.


--
-- Data for Name: classes; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.classes (class_id, name, number_of_students) FROM stdin;
1	Class 4	5
5	Class 5	5
3	Class 3	5
2	Class 2	5
4	Class 1	5
\.


--
-- Data for Name: questions; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.questions (question_id, chapter_id, topic_id, question) FROM stdin;
1	1	1	Is a cube also a prism?
2	1	1	What is the difference between a prism and a pyramid?
3	1	1	What shape of face is common to all the prisms?
4	1	1	What other shapes do these prisms have?
5	1	1	How many such faces each?
6	1	1	What shape of face is common to all the pyramids?
7	1	1	All the triangular faces meet at point.
8	1	1	Identify any other shape in each of the pyramids.
9	1	1	Sort 3D shapes by the number of flat faces. Write their names here.
10	1	1	Sort 3D shapes by the number of straight edges. Write their names here.
11	1	1	Identify any relationship that you may find between the number of faces (F), edges (E), and corners (V). Calculate F+V–E in each case. What do you notice?
12	1	1	Can you construct a 3D shape with 3 flat faces?
13	1	1	Which of these nets can be folded to make a solid of the kind given below?
14	1	1	Nitesh cuts up a net on the folds. Here are its pieces. Which solid has the above pieces in its net?
15	1	2	What parts of the building have you shown in your model (for example, roof, pillars, base, etc.)?
16	1	2	Why did you select these parts?
17	1	2	What shapes will model these parts well?
18	1	2	How is your model similar to the picture of the real building?
19	1	2	How is it different from the real building?
20	1	2	Do you think it looks like the Qutub Minar?
21	1	2	What shape would you use if you made a model of the Qutub Minar? Why?
22	1	2	How many such shapes will you use?
23	1	2	What is common to all of these bricks?
24	1	3	How many angles are there in this boat drawing?
25	1	3	Where do you see angles in the classroom? Give a few examples.
26	1	3	Write the names of objects where you find right angles.
27	1	3	Identify the angles that you think are right angles and circle them in the dot grid given below. Check using your right angle checker.
28	1	3	Name some objects from your classroom which have an acute angle.
29	1	3	Name some objects from your classroom which have an obtuse angle.
30	1	3	Identify all angles in the following letters.
31	1	3	What kinds of angles does a triangle have?
32	1	3	What kinds of angles do you see in the rectangle?
33	1	3	What has happened to the angles of the new shape? Are they still right angles? What types of angles have been formed?
34	1	3	Similarly, push one side of a square. Are they still right angles? What types of angles have been formed?
35	1	3	How are the angles of triangles and rectangles similar or different?
36	1	3	What relation do you notice between the number of sides and the number of angles?
37	1	3	Mark the right angles and write the number of right angles in each figure.
38	1	3	Which of the above shapes have only right angles?
39	1	3	Identify the shape that has: 2 right angles, 1 acute, and 1 obtuse angle.
40	1	3	Identify the shape that has: 1 right, 2 obtuse, and 1 acute angle.
41	1	3	Identify the shape that has: 2 obtuse, and 2 acute angles.
42	1	3	Identify the shape that has: 4 right angles.
43	1	4	Does the shape of the triangle change if we gently push one of its sides?
44	1	4	Does the shape of the rectangle change if we gently push one of its sides?
45	1	4	What shapes did you make?
46	1	4	How many shapes have you made with 1 right angle?
47	1	4	How many shapes have you made with 2 right angles?
48	1	4	How many shapes have you made with 3 right angles?
49	1	4	How many shapes have you made with all right angles?
50	1	4	In what ways are rectangle and square different from these shapes?
51	1	4	Draw a 2D shape that has less than 5 angles.
52	1	4	Draw a 2D shape with more than 5 angles.
53	1	5	Can you make a circle using straws?
54	1	5	What will happen if we take straws of unequal lengths?
55	1	5	Can you use a scale to draw a circular shape? Let us see.
56	1	5	What do you get?
57	1	5	Are these right angles?
58	1	5	Discuss where the centre is. Do you notice that all the diameters pass through the centre?
59	1	5	Discuss if there is any relationship between the radius and the diameter of a circle.
60	1	5	Name the wheel with the longest radius.
61	1	5	Name the wheel with the shortest radius.
62	1	5	Name the wheel with the longest diameter.
63	1	5	Name the wheel with the shortest diameter.
64	1	6	What is the face opposite to the face numbered 2?
65	1	6	What is the face opposite to the face numbered 3?
66	1	6	What is the face opposite to the face numbered 4?
67	1	6	What colour is the face that is opposite to the red face?
68	1	6	What colour is the face that is opposite to the yellow face?
69	1	6	Which faces have common edges with the face numbered 1?
70	1	6	Which face has no common edge with the face numbered 1?
71	1	6	In which circle did you write triangular prism and rectangular pyramid?
72	1	6	How many cubes are there in each of these cube towers?
73	1	6	Can you complete the following cubes?
74	1	6	Match the pictures to the descriptions and name the shapes.
75	1	6	Each one is different. How? Discuss.
76	1	6	Identify the hidden shapes and write their names.
77	1	6	Draw 2 lines to divide the triangle into 1 square and 2 triangles.
78	1	6	Draw 2 lines to divide the square into 3 triangles.
79	1	6	Draw lines to show the cuts needed on the shapes in the left column to get the smaller shapes on the right.
80	1	6	How many rectangles can you see in his web?
81	1	6	Can he begin at point A and leave from point B without walking on any wall more than once?
82	1	6	How many triangles are in her web?
83	1	6	Can she begin at point A and reach back to the same point without walking on any wall more than once?
84	2	7	Q: 1. Look at the picture and answer the following: a) Which game are the children playing? _________ b) Who is looking from the top?___________________ c) In Scene 1, if Rani faces towards the hut, will she be able to see who all are hiding near the hut? ___________________
85	2	7	Q: In Scene 4, can Mini see all the children playing the game? Discuss.
86	2	7	Q: Whose drawing shows the following views? View of the Brick drawing Name of the child The top view The front view The side view
87	2	7	Q: 3. Look at the pictures and name the objects. Also write which view of the object is given. Name _____________________ View ______________________
88	2	7	Q: 4. Jagat and Rani have made different drawings of the same objects. Match the views with the objects.
89	2	8	Q: 1. Mark Jagat’s position in the picture.
90	2	8	Q: 2. Describe the position of the blue bag _____________________.
91	2	8	Q: 3. What do you see on the middle desk of the second row? _____________________.
92	2	8	Q: 4. Where is the notebook kept — the first desk in the second row or the middle desk in the third row?
93	2	8	Q: 5. Draw an apple on the third desk of the second row.
94	2	9	Q: How many routes were you able to find? ________ (You may use different colour pencils to trace the different routes)
95	2	9	Q: Which is the shortest route? How do you know?
96	2	9	Q: The water delivery man has turned left from the entrance. Help him reach MDM Kitchen by telling him the route. Write the directions below.
97	2	9	Q: Rajat is not feeling well. Which way will you choose to take him to the medical room from the library?
98	2	9	Q: After the assembly in the playground, Bholu must go to the IT room and Rani has to go to the sports room. Trace their paths. Which way is longer?
99	3	10	Number of cups = ____________
100	3	10	How many coconut trees does Gundappa have? ___________
101	3	10	How do you know?
102	3	10	How many coconuts has he plucked?________
103	3	10	How many coconut laddoos are there in the trays? ____________
104	3	10	How many milk pedas are there in the trays? ____________
105	3	10	How much money? ________
106	3	10	How much money? ________
107	3	10	How did you count them? Discuss in class.
108	3	11	Describe Shiv’s arrangement and write his numbers.
109	3	11	Describe Shirley’s arrangement and write her numbers.
110	3	11	Do you think all numbers in the times-2 table are even?
111	3	11	Which numbers are even and which are odd? Discuss.
112	3	11	Explore your textbook and find out what Shirley has seen. Draw a square on the even numbers. Put a circle on the odd numbers.
113	3	11	What do you think? Check and discuss.
114	3	11	Choose any 10 numbers in order without skipping any (consecutive numbers). Write whether they are even or odd below each number. What do you notice? Discuss.
115	3	11	Identify which of the following numbers are even and which are odd. Explain your reasoning.
116	3	11	Odd numbers : __________________________________________________
117	3	11	Even numbers : __________________________________________________
118	3	11	Make two 2-digit numbers using the digits 1 and 6 without repetition.
119	3	11	Identify the numbers as even or odd. Now choose any two digits and make 2-digit numbers in such a way that the numbers are even.
120	3	11	Are there more even or odd numbers between 1 and 100?
\.


--
-- Data for Name: students; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.students (student_id, name, photo_id, class_id, attendance) FROM stdin;
1	Kkrishna	photo_2989	1	99
2	Divyansh	photo_1200	1	96
3	Siddhant	photo_6864	1	89
4	Harshal	photo_1857	1	91
5	Yash	photo_2715	1	93
6	Diya Shah	photo_8288	2	99
7	Vihan Patel	photo_1032	2	98
8	Priya Jain	photo_1855	2	96
9	Nikhil Kumar	photo_6164	2	85
10	Nikhil Shah	photo_5740	2	96
11	Dev Singh	photo_7324	3	89
12	Priya Agarwal	photo_9746	3	89
13	Dev Agarwal	photo_1322	3	100
14	Aditi Agarwal	photo_8553	3	86
15	Nikhil Agarwal	photo_8895	3	85
16	Tanvi Kumar	photo_9438	4	87
17	Aarav Sharma	photo_3478	4	90
18	Aarav Jain	photo_9442	4	97
19	Diya Sharma	photo_2643	4	95
20	Rohan Gupta	photo_6698	4	94
21	Vihan Agarwal	photo_5682	5	94
22	Priya Jain	photo_4881	5	88
23	Arjun Patel	photo_5864	5	93
24	Meera Patel	photo_8834	5	98
25	Aarav Patel	photo_1925	5	97
\.


--
-- Data for Name: subjects; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.subjects (subject_id, name, class_id, number_of_topics) FROM stdin;
1	Mathematics	1	11
\.


--
-- Data for Name: topics; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.topics (topic_id, name, subject_id, chapter_id) FROM stdin;
1	3D Shapes and Their Properties	1	1
2	Model Building and Representation	1	1
3	Angles and Their Types	1	1
4	2D Shapes and Their Properties	1	1
5	Circle and Its Properties	1	1
6	Spatial Reasoning and Visualization	1	1
7	Views and Perspectives	1	2
8	Grid and Positioning	1	2
9	Pathfinding and Directions	1	2
10	Counting and Arrangements	1	3
11	Even and Odd Numbers	1	3
\.


--
-- Data for Name: weak_topics; Type: TABLE DATA; Schema: public; Owner: -
--

COPY public.weak_topics (topic_id, student_id, weakness) FROM stdin;
1	1	0
1	2	0
1	3	0
1	4	0
1	5	0
2	1	0
2	2	0
2	3	0
2	4	0
2	5	0
3	1	0
3	2	0
3	3	0
3	4	0
3	5	0
4	1	0
4	2	0
4	3	0
4	4	0
4	5	0
5	1	0
5	2	0
5	3	0
5	4	0
5	5	0
6	1	0
6	2	0
6	3	0
6	4	0
6	5	0
7	1	0
7	2	0
7	3	0
7	4	0
7	5	0
8	1	0
8	2	0
8	3	0
8	4	0
8	5	0
9	1	0
9	2	0
9	3	0
9	4	0
9	5	0
10	1	0
10	2	0
10	3	0
10	4	0
10	5	0
11	1	0
11	2	0
11	3	0
11	4	0
11	5	0
\.


--
-- Name: chapters_chapter_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.chapters_chapter_id_seq', 3, true);


--
-- Name: classes_class_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.classes_class_id_seq', 5, true);


--
-- Name: questions_question_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.questions_question_id_seq', 120, true);


--
-- Name: students_student_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.students_student_id_seq', 25, true);


--
-- Name: subjects_subject_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.subjects_subject_id_seq', 3, true);


--
-- Name: topics_topic_id_seq; Type: SEQUENCE SET; Schema: public; Owner: -
--

SELECT pg_catalog.setval('public.topics_topic_id_seq', 11, true);


--
-- Name: chapters chapters_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.chapters
    ADD CONSTRAINT chapters_pkey PRIMARY KEY (chapter_id);


--
-- Name: classes classes_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.classes
    ADD CONSTRAINT classes_pkey PRIMARY KEY (class_id);


--
-- Name: questions questions_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions
    ADD CONSTRAINT questions_pkey PRIMARY KEY (question_id);


--
-- Name: students students_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_pkey PRIMARY KEY (student_id);


--
-- Name: subjects subjects_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.subjects
    ADD CONSTRAINT subjects_pkey PRIMARY KEY (subject_id);


--
-- Name: topics topics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics
    ADD CONSTRAINT topics_pkey PRIMARY KEY (topic_id);


--
-- Name: weak_topics weak_topics_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weak_topics
    ADD CONSTRAINT weak_topics_pkey PRIMARY KEY (topic_id, student_id);


--
-- Name: idx_chapters_vector_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_chapters_vector_id ON public.chapters USING btree (vector_id);


--
-- Name: idx_questions_chapter_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_questions_chapter_id ON public.questions USING btree (chapter_id);


--
-- Name: idx_questions_topic_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_questions_topic_id ON public.questions USING btree (topic_id);


--
-- Name: idx_students_class_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_students_class_id ON public.students USING btree (class_id);


--
-- Name: idx_subjects_class_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_subjects_class_id ON public.subjects USING btree (class_id);


--
-- Name: idx_topics_chapter_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_topics_chapter_id ON public.topics USING btree (chapter_id);


--
-- Name: idx_topics_subject_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_topics_subject_id ON public.topics USING btree (subject_id);


--
-- Name: idx_weak_topics_student_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_weak_topics_student_id ON public.weak_topics USING btree (student_id);


--
-- Name: idx_weak_topics_topic_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_weak_topics_topic_id ON public.weak_topics USING btree (topic_id);


--
-- Name: questions questions_chapter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions
    ADD CONSTRAINT questions_chapter_id_fkey FOREIGN KEY (chapter_id) REFERENCES public.chapters(chapter_id) ON DELETE CASCADE;


--
-- Name: questions questions_topic_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.questions
    ADD CONSTRAINT questions_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES public.topics(topic_id) ON DELETE CASCADE;


--
-- Name: students students_class_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.students
    ADD CONSTRAINT students_class_id_fkey FOREIGN KEY (class_id) REFERENCES public.classes(class_id) ON DELETE CASCADE;


--
-- Name: subjects subjects_class_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.subjects
    ADD CONSTRAINT subjects_class_id_fkey FOREIGN KEY (class_id) REFERENCES public.classes(class_id) ON DELETE CASCADE;


--
-- Name: topics topics_chapter_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics
    ADD CONSTRAINT topics_chapter_id_fkey FOREIGN KEY (chapter_id) REFERENCES public.chapters(chapter_id) ON DELETE CASCADE;


--
-- Name: topics topics_subject_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.topics
    ADD CONSTRAINT topics_subject_id_fkey FOREIGN KEY (subject_id) REFERENCES public.subjects(subject_id) ON DELETE CASCADE;


--
-- Name: weak_topics weak_topics_student_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weak_topics
    ADD CONSTRAINT weak_topics_student_id_fkey FOREIGN KEY (student_id) REFERENCES public.students(student_id) ON DELETE CASCADE;


--
-- Name: weak_topics weak_topics_topic_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.weak_topics
    ADD CONSTRAINT weak_topics_topic_id_fkey FOREIGN KEY (topic_id) REFERENCES public.topics(topic_id) ON DELETE CASCADE;


--
-- Name: SCHEMA public; Type: ACL; Schema: -; Owner: -
--

GRANT ALL ON SCHEMA public TO cloudsqlsuperuser;


--
-- PostgreSQL database dump complete
--

