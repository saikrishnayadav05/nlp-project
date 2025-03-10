# -*- coding: utf-8 -*-
"""7/1/25 Assignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1umESqlecYvFp7olMZSTLF8xG-YWEdCG8
"""

#Write a Python program to calculate the sum of all even numbers between 1 and a given positive integer n
def sum_of_even_numbers(n):
    # Initialize sum to 0
    total = 0
    for i in range(2, n+1, 2):
        total += i

    return total

# Input: positive integer n
n = int(input("Enter a positive integer: "))

# Check if n is positive
if n > 0:
    result = sum_of_even_numbers(n)
    print(f"The sum of all even numbers between 1 and {n} is: {result}")
else:
    print("Please enter a positive integer.")

