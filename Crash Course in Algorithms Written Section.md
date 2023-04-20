# Dynamic Programming

## Content Table
- [Dynamic Programming](#dynamic-programming)
  - [Content Table](#content-table)
  - [Theory of Dynamic programming](#theory-of-dynamic-programming)
    - [Definition of Dynamic Programming](#definition-of-dynamic-programming)
    - [Benefits of DP](#benefits-of-dp)
  - [Step in solving DP problem](#step-in-solving-dp-problem)
    - [Identify the problem as a DP problem](#identify-the-problem-as-a-dp-problem)
    - [Define subproblems and find recursive relations](#define-subproblems-and-find-recursive-relations)
    - [Formulate the recursive relation:](#formulate-the-recursive-relation)
    - [Implement the DP solution:](#implement-the-dp-solution)
  - [Examples](#examples)
    - [Fibonacci sequence](#fibonacci-sequence)
      - [Analysis](#analysis)
      - [Top-down approach with memoization](#top-down-approach-with-memoization)
      - [Bottom-up approach with a table](#bottom-up-approach-with-a-table)
      - [Time and Space Complexity](#time-and-space-complexity)
    - [Longest common subsequence](#longest-common-subsequence)
      - [Analysis](#analysis-1)
      - [Top-down approach with memoization](#top-down-approach-with-memoization-1)
      - [Bottom-up approach with a table](#bottom-up-approach-with-a-table-1)
      - [Time and Space Complexity](#time-and-space-complexity-1)
  - [Conclusion](#conclusion)
    - [Recap of DP](#recap-of-dp)
  - [Quiz Questions about dynamic programming](#quiz-questions-about-dynamic-programming)
## Theory of Dynamic programming



### Definition of Dynamic Programming
Dynamic programming, developed by Richard Bellman in 1950, is both a mathematical optimization method and a computer programming method. The main purpose of this solution is to find the optimal substructure. It involves breaking a larger problem into smaller and smaller subproblems and then building up correct solutions to larger and larger subproblems. 
<!-- ## When and Why to use DP

### When to use DP
 
### Why to use DP -->

### Benefits of DP
1. Efiiciency: DP will stored the previous results in a table and use them to solve the current problem. This will save a lot of time and space.
2. Optimal solution:The main purpose of DP is to find the optimal substructure. So DP will find the optimal solution for the problem.
3. Better Understanding of the problem: DP will break the problem into smaller subproblems. By using memoization, it becomes easier to check previous results, which can help beginners understand how it works and debug more effectively. This will help us to understand the problem better.
## Step in solving DP problem

### Identify the problem as a DP problem
- Typically, most of Dynamic programming are required to find the optimal solution for a problem.
- Almost all Dynamic programming problems satisfy overlapping subproblems property.

### Define subproblems and find recursive relations
 Choose a way to represent the subproblems using a state, which is a set of variables that uniquely define a subproblem. This state representation will allow you to map the subproblems to their corresponding solutions.
### Formulate the recursive relation:
Establish a relationship between the solution of a subproblem and the solutions of its smaller subproblems. This is typically done using a recursive function, where the function calls itself with smaller inputs. Identify the base cases, which are the smallest subproblems that can be solved directly.

### Implement the DP solution:
There are two main approaches to implement the DP solution: top-down (memoization) and bottom-up (tabulation).

a. Top-down (memoization):
Start solving the original problem by breaking it into subproblems, and then recursively solve these subproblems. Store the results of each subproblem in a data structure (like an array or dictionary) to avoid redundant computations. Look up the solutions to already-solved subproblems when needed.

b. Bottom-up (tabulation):
Solve the subproblems iteratively, starting from the smallest ones, and use a table to store their solutions. Build the solution to the original problem by combining the solutions of smaller subproblems in the correct order.

## Examples

### Fibonacci sequence
The Fibonacci sequence is a series of numbers in which each number is the sum of the two preceding ones, usually starting with 0 and 1. The sequence goes: 0, 1, 1, 2, 3, 5, 8, 13, 21, and so on. Mathematically, the Fibonacci sequence can be defined using the following recurrence relation:

F(n) = F(n-1) + F(n-2)

with the initial conditions:

F(0) = 0
F(1) = 1
#### Analysis
1. Overlapping subproblems:
In the Fibonacci sequence problem, to calculate F(n), you need the results of F(n-1) and F(n-2). Similarly, to calculate F(n-1), you need F(n-2) and F(n-3). As you can see, the subproblems F(n-2) and F(n-3) are being solved multiple times when computing the Fibonacci numbers. These overlapping subproblems are a key characteristic of dynamic programming problems.
2. Optimal substructure:
Optimal substructure refers to the property that an optimal solution to the problem can be constructed by combining the optimal solutions of its subproblems. The Fibonacci sequence problem exhibits optimal substructure because the solution for F(n) can be found by using the solutions for F(n-1) and F(n-2). In other words, if you know the optimal solutions to F(n-1) and F(n-2), you can find the optimal solution to F(n) by simply adding them together.
3. For the Fibonacci sequence problem, the subproblems and recursive relationship can be defined as follows:

- Subproblems:
The goal is to find the nth Fibonacci number, F(n). To achieve this, we can break the problem down into smaller subproblems: finding the (n-1)th and (n-2)th Fibonacci numbers, F(n-1) and F(n-2). This process can be applied recursively until we reach the base cases F(0) and F(1).

- Recursive relationship:
The recursive relationship for the Fibonacci sequence is based on the fact that each Fibonacci number is the sum of the two preceding ones. Mathematically, it can be expressed as:

`F(n) = F(n-1) + F(n-2)`

  - Base cases:
The base cases for the Fibonacci sequence are the first two numbers in the sequence, which are given as:

```
F(0) = 0
F(1) = 1
```

These base cases are used to stop the recursion and build the solution for higher Fibonacci numbers.

So, for the Fibonacci sequence problem, the subproblems are finding F(n-1) and F(n-2), and the recursive relationship is F(n) = F(n-1) + F(n-2). By solving these subproblems recursively and combining their solutions using the recursive relationship, we can find the nth Fibonacci number efficiently using dynamic programming techniques like memoization or tabulation.

#### Top-down approach with memoization
Here's a Python implementation using memoization:
```python
def fib_memoization(n):
    if n <= 1:
        return n
    memo = {}
    if n not in memo:
        memo[n] = fib_memoization(n - 1) + fib_memoization(n - 2)

    return memo[n]
```
Here's a Java implementation using memoization:
```java
public static int fib(int n ){
        if (n <= 1 ) {
            return n;
        }
        List<Integer> fib= new ArrayList<Integer>();
        fib.add(0);
        fib.add(1);
        for (int i = 2; i <= n; i++) {
            fib.add(fib.get(i-1)+fib.get(i-2));
        }
        return fib.get(n-1);
    }
```
#### Bottom-up approach with a table
Here's a Python implementation using tabulation:

```python
def fib_tabulation(n):
    if n <= 1:
        return n

    fib = [0] * (n + 1)
    fib[1] = 1

    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]

    return fib[n]

n = 10  # Desired Fibonacci number
print(fib_tabulation(n))
```

#### Time and Space Complexity
1. Memoization (Top-Down) 
- Time complexity: The time complexity of the memoized Fibonacci solution is O(n). This is because each Fibonacci number is calculated once and stored in the memo dictionary. The function is called for each number from 2 to n, making n-1 calls, but it performs a constant amount of work for each call (looking up in the memo dictionary), which results in a linear time complexity.

- Space complexity:The space complexity of the memoized solution is also O(n). This is due to the memo dictionary, which can store up to n elements (from F(2) to F(n)). Additionally, there is an overhead of recursion, which can result in a call stack depth of up to n. So, the space complexity for the memoized solution is O(n) due to both the memo dictionary and the recursion call stack.
2. Tabulation (Bottom-Up)
Time complexity:
The time complexity of the tabulated Fibonacci solution is O(n). In this case, the Fibonacci numbers are computed iteratively from F(2) to F(n) using a loop. Since there are n-1 iterations in the loop, and each iteration takes constant time, the overall time complexity is linear.

Space complexity:
The space complexity of the tabulated solution is O(n) due to the use of the fib array to store the computed Fibonacci numbers. However, note that there is no recursion in this approach, so there is no additional overhead from a call stack. The space complexity can be reduced to O(1) by only keeping track of the last two Fibonacci numbers in the sequence, as shown below:
```python
def fib_tabulation_optimized(n):
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b

```
### Longest common subsequence
Longest Common Subsequence https://leetcode.com/problems/longest-common-subsequence/

Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.
A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.
For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.
Example 1:
```
Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
```
Example 2:
```
Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
```
Example 3:
```
Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
```
Constraints:
`1 <= text1.length, text2.length <= 1000`
`text1` and `text2` consist of only lowercase English characters.

#### Analysis

#### Top-down approach with memoization
  In this solution, we use a helper function lcs_helper that takes the two strings, their current lengths m and n, and a memo table to store the results of previously computed subproblems. The function first checks if either string is empty, in which case the longest common subsequence is 0. If the result has been previously calculated, it returns the memoized value. Otherwise, it calculates the result using recursion and stores it in the memo table.
 ```python
 class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        memo = [[-1] * (n + 1) for _ in range(m + 1)]
        return self.lcs_helper(text1, text2, m, n, memo)

    def lcs_helper(self, text1: str, text2: str, m: int, n: int, memo: list) -> int:
        # Base case: if either string is empty, the longest common subsequence is 0
        if m == 0 or n == 0:
            return 0

        # If the result is already calculated, return the memoized value
        if memo[m][n] != -1:
            return memo[m][n]

        # If the characters at the current positions match, increment the subsequence length and continue the recursion
        if text1[m - 1] == text2[n - 1]:
            memo[m][n] = 1 + self.lcs_helper(text1, text2, m - 1, n - 1, memo)
        else:
            # If the characters don't match, find the maximum length between two possible options
            memo[m][n] = max(self.lcs_helper(text1, text2, m - 1, n, memo),
                             self.lcs_helper(text1, text2, m, n - 1, memo))

        return memo[m][n]
 ```
#### Bottom-up approach with a table
To solve this problem, you can use dynamic programming. Create a 2D table dp, where dp[i][j] represents the length of the longest common subsequence between text1[0...i-1] and text2[0...j-1]. Then, you can iteratively build the table using the following rules:
1. If text1[i-1] == text2[j-1], then the length of the longest common subsequence for this pair of characters is dp[i-1][j-1] + 1.
2. If text1[i-1] != text2[j-1], then the length of the longest common subsequence for this pair of characters is max(dp[i-1][j], dp[i][j-1]).
  ```python
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]
  ```

#### Time and Space Complexity
1. Bottom-up solution:
  - Time Complexity: O(m * n)
In the bottom-up approach, we fill the dp table iteratively with two nested loops, iterating through all the elements in text1 and text2. The dimensions of the table are (m+1) x (n+1), where m and n are the lengths of text1 and text2, respectively. Thus, the time complexity is O(m * n).

  - Space Complexity: O(m * n)
The space complexity of the bottom-up solution is determined by the size of the dp table, which has (m+1) x (n+1) elements. Therefore, the space complexity is O(m * n).

2. Top-down solution:
- Time Complexity: O(m * n)
The top-down approach uses memoization to avoid redundant calculations. In the worst case, we would need to calculate the longest common subsequence for each unique combination of indices (i, j) in the two strings. Since there are m * n unique combinations, the time complexity is O(m * n).

- Space Complexity: O(m * n)
The space complexity of the top-down solution comes from the size of the memo table and the depth of the recursion call stack. The memo table has dimensions (m+1) x (n+1), contributing O(m * n) space complexity. The maximum depth of the recursion call stack is min(m, n), in the worst case, when the strings have no common characters. However, this contribution to the space complexity is usually less significant than the memo table, so we can approximate the space complexity as O(m * n).



## Conclusion

### Recap of DP
- Dynamic Programming (DP) is a powerful problem-solving technique used in computer science, mathematics, and other fields to solve complex problems by breaking them down into simpler, overlapping subproblems. This method is particularly useful when a problem exhibits two key properties: optimal substructure and overlapping subproblems.

  - Optimal Substructure: A problem has an optimal substructure if the optimal solution to the problem can be constructed from the optimal solutions of its subproblems. In other words, we can break the problem down into smaller parts and find the best solution for each part, then combine those solutions to solve the original problem.

  - Overlapping Subproblems: A problem has overlapping subproblems if the same subproblem is solved multiple times. By storing the solution to a subproblem once it is solved (a process called memoization), we can avoid redundant computations, resulting in significant time savings.

- Dynamic Programming can be applied using two main approaches:

   1. Top-down (Memoization): In the top-down approach, the problem is solved by breaking it down into smaller subproblems, and the solutions are stored in a data structure, such as an array or a hash table, for future reference. When a subproblem is encountered again, its solution is retrieved from the data structure, avoiding redundant computation.

  2. Bottom-up (Tabulation): In the bottom-up approach, the problem is solved by iteratively building a table of solutions for smaller subproblems, starting from the smallest and working up to the main problem. This approach does not require recursion and is typically more efficient in terms of space complexity.

## Quiz Questions about dynamic programming
1. What is the primary purpose of dynamic programming in problem-solving?
a) Divide and conquer
b) Optimization
c) Backtracking
d) Greedy algorithms
B
2. Which of the following is NOT a key characteristic of dynamic programming problems?
a) Overlapping subproblems
b) Optimal substructure
c) Greedy choice property
d) Memoization
C
3. What is the technique used to store solutions of subproblems in dynamic programming called?
a) Caching
b) Tabulation
c) Memoization
d) Serialization
C
4. In the context of dynamic programming, what does the term "bottom-up approach" refer to?
a) Solving larger subproblems before smaller ones
b) Solving smaller subproblems before larger ones
c) Starting from the middle of a problem and expanding outward
d) None of the above
B
5. What is the primary difference between dynamic programming and recursive approaches to solving problems?
a) Dynamic programming is always faster than recursion
b) Dynamic programming does not use recursion
c) Dynamic programming avoids redundant calculations by reusing previously computed results
d) Dynamic programming only works on optimization problems
C
6. In the context of dynamic programming, what does the term "state" refer to?
a) The current position in the problem space
b) The set of variables that define a subproblem
c) The current value of the objective function
d) The optimal solution to a problem
B
7. What is the time complexity of the Fibonacci sequence when solved using dynamic programming with memoization?
a) O(2^n)
b) O(n^2)
c) O(n)
d) O(log n)
C
8. What is the primary difference between the top-down and bottom-up approaches in dynamic programming?
a) Top-down starts by solving smaller subproblems, while bottom-up starts by solving larger subproblems
b) Top-down starts by solving larger subproblems, while bottom-up starts by solving smaller subproblems
c) Top-down uses memoization, while bottom-up uses tabulation
d) Top-down uses tabulation, while bottom-up uses memoization
C
9. In the context of dynamic programming, which technique is used to build a solution iteratively, without using recursion?
a) Memoization
b) Tabulation
c) Iterative deepening
d) Branch and bound
B
10. Consider the following 0/1 Knapsack problem: You are given a list of items with their respective weights and values, and a knapsack with a maximum weight capacity. The items can only be taken in their entirety or not taken at all. What is the maximum value that can be obtained from the given items without exceeding the knapsack's weight limit?
Items:

    Weight: 10 kg, Value: 60 USD
    Weight: 20 kg, Value: 240 USD
    Weight: 30 kg, Value: 100 USD
    Weight: 40 kg, Value: 140 USD
    Weight: 50 kg, Value: 200 USD
    Weight: 60 kg, Value: 240 USD

    Knapsack maximum weight capacity: 80 kg

    a) 420 USD
    b) 440 USD
    c) 460 USD
    d) 480 USD
    D



