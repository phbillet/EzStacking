DELETE FROM model
WHERE name = 'iris'
AND version = 1;

DELETE FROM eda
WHERE name = 'iris'
AND version = 1;

DELETE FROM solution
WHERE name = 'iris'
AND version = 1;

DELETE FROM problem
WHERE name = 'iris';

INSERT INTO problem (name, path , type, target)
VALUES('iris', '/home/philippe/development/python/EZStacking/dataset/Iris.csv','classification', 'Species');


SELECT *
FROM problem;

INSERT INTO solution (name, version, correlation, nb_model, nb_feature, score, test_size)
VALUES('iris', 1, 0.75, 5, 5, 0.7, 0.33);


SELECT *
FROM solution;

INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)
VALUES('iris', 1, 'SepalLengthCm', 'num', '[4.3, 7.9]', 0, 0, 0);
INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)
VALUES('iris', 1, 'SepalWidthCm', 'num', '[2.0, 4.4]', 0, 0, 0);
INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)
VALUES('iris', 1, 'PetalLengthCm', 'num', '[1.0, 6.9]', 0, 0, 0);
INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)
VALUES('iris', 1, 'PetalWidthCm', 'num', '[0.1, 2.5]', 0, 0, 0);
INSERT INTO eda (name, version, feature, type, range, drop_user, drop_correlation, target)
VALUES('iris', 1, 'Species' , 'num', NULL, 0, 0, 1);

INSERT INTO model (name, version, step)
VALUES ('iris', 1, 1);
INSERT INTO model (name, version, step)
VALUES ('iris', 1, 2);
INSERT INTO model (name, version, step)
VALUES ('iris', 1, 3);

SELECT *
FROM problem 
INNER JOIN solution 
ON problem.name = solution.name