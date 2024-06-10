Sure, here's a Python function to check if a given string is a palindrome:

```python
def is_palindrome(s):
    """
    Checks if a given string is a palindrome.
    
    Args:
        s (str): The string to be checked.
        
    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    # Remove non-alphanumeric characters and convert to lowercase
    clean_s = ''.join(char.lower() for char in s if char.isalnum())
    
    # Check if the cleaned string is equal to its reverse
    return clean_s == clean_s[::-1]
```

This function takes a string `s` as input and returns `True` if the string is a palindrome, and `False` otherwise.

Here's how it works:

1. The function first removes all non-alphanumeric characters (e.g., spaces, punctuation marks) from the input string `s` using a list comprehension and the `isalnum()` method. It also converts all characters to lowercase using the `lower()` method. The resulting cleaned string is stored in the `clean_s` variable.

2. The function then checks if `clean_s` is equal to its reverse, which is obtained using the slice notation `[::-1]`. This reverses the string by taking every element from start to end, but in reverse order.

3. If `clean_s` is equal to its reverse, the function returns `True`, indicating that the input string is a palindrome. Otherwise, it returns `False`.

Here are some examples of how to use the `is_palindrome()` function:

```python
print(is_palindrome("racecar"))     # Output: True
print(is_palindrome("A man a plan a canal Panama"))  # Output: True
print(is_palindrome("hello"))       # Output: False
print(is_palindrome("Python"))      # Output: False
```

Note that the function considers only alphanumeric characters when checking for palindromes, ignoring non-alphanumeric characters and case sensitivity.