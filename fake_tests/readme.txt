copy fake_tests to same folder of ex.py:

    |
    |---fake_tests
        |---readme.txt
        |---created
        |---original
        |---out
    |---ex2.py


to run:
python fake_tests/fake_it.py 


customize:
    run one of this:
        test(test_fake_files=False):
            - you're testing the original submit files.
            - number_of_files - number of testings
        test(test_fake_files=True):
            - you're testing the new random created files (located in "created")
            - number_of_files - number of files to create (and also number of testings)

After running all of the tests, you'll get an average report.

green text - more than 93%
red text - less than 93%


there're 30 fake files created by me, if you want to create more:
pip install sklearn
uncomment "create_files()"
and of course set: test(test_fake_files=True)
