#!/usr/bin/env python
# coding: utf-8

"""Assign reviewers to be even within each group."""

import argparse
import collections
import copy
import random

import numpy as np
import pandas as pd


def parse_args():
    """Command-line parsing."""
    parser = argparse.ArgumentParser(description="Assign reviewer for b4lecture.")
    parser.add_argument("--users", type=str, default="./users.csv")
    args = parser.parse_args()
    return args


def get_list_from_df(df, query_column, query, return_column):
    """
    Search any column from the data frame for rows matching the query.

    Return a list of corresponding target columns.

    Args:
        df (pandas.core.frame.DataFrame): Dataframe.
        query_column (str): Column name to be searched.
        query (str): Search query.
        return_column (str): Column name to be returned.

    Returns:
        target_list (List[str]): List of search results.
    """
    df_extracted = df.query(f'{query_column} in ["{query}"]')
    target_list = list(df_extracted[return_column])
    return target_list


def convert_value_with_correspondence_lists(value, list_from, list_to):
    """
    Convert values based on two lists.

    Args:
        value (str | int | ...): Value to be converted.
        list_from (List[N]): List of converted sources.
        list_to (List[N]): List of conversion targets.

    Returns:
        value_converted (str | int | ...): Converted value.

    Notes:
        The two lists list_from and list_to must have the same length.

    Examples:
        >>> value = 'a'
        >>> list_from = ['a', 'b', 'c']
        >>> list_to = [1, 2, 3]
        >>> converted_value \
            = convert_value_with_correspondence_lists(value, list_from, list_to)
        >>> print(converted_value)  # 1
    """
    assert len(list_from) == len(
        list_to
    ), f"The lengths of the two lists differ ({len(list_from)} and {len(list_to)})."
    list_from = list_from + ["Unassigned", -1]
    list_to = list_to + [-1, "Unassigned"]
    value_converted = list_to[list_from.index(value)]
    return value_converted


def convert_list_with_correspondence_lists(target_list, list_from, list_to):
    """
    Convert a list based on two lists.

    Args:
        target_list (List[str | int | ...]): List to be converted.
        list_from (List, len=N): List of converted sources.
        list_to (List, len=N): List of conversion targets.

    Returns:
        target_list_converted (List[str | int | ...]): Converted list.

    Notes:
        The two lists list_from and list_to must have the same length.

    Examples:
        >>> target_list = ['c', 'b']
        >>> list_from = ['a', 'b', 'c']
        >>> list_to = [1, 2, 3]
        >>> target_list_converted \
            = convert_value_with_correspondence_lists(target_list, list_from, list_to)
        >>> target_list_converted
            # [3, 2]
    """
    target_list_converted = [
        convert_value_with_correspondence_lists(value, list_from, list_to)
        for value in target_list
    ]
    return target_list_converted


def convert_assigntable_reviewee_to_reviewer(reviewees, reviewers, assign_history):
    """
    Convert assign table from the reviewee's view to the reviewer's view.

    Args:
        reviewees (List, len=M): The list of reviewees number.
        reviewers (List, len=N): The list of reviewers number.
        assign_history (numpy.array, shape=(EX, M)): Reviewer assignment's history.

    Returns:
        table_converted (numpy.array, shape=(EX, N)): Converted reviewer assignment's history.

    Notes:
        The length of list reviewees and assign_history.shape[1] must be equal.

    Examples:
        >>> reviewees = [1, 2, 3]
        >>> reviewers = [11, 12, 13, 14, 15]
        >>> assign_history = np.array([
                [11, 12, 13],
                [14, 15, 11],
                [12, 13, 14],
                [15, 11, 12]
            ])
        >>> convert_assigntable_reviewee_to_reviewer(reviewees, reviewers, assign_history)
            # np.array([
            #   [1, 2, 3, -1, -1],
            #   [3, -1, -1, 1, 2],
            #   [-1, 1, 2, 3, -1],
            #   [2, 3, -1, -1, 1]
            # ])
    """
    assert (
        len(reviewees) == assign_history.shape[1]
    ), "Mismatch the number of reviewees."
    lecture_total_number = assign_history.shape[0]
    table_converted = np.full((lecture_total_number, len(reviewers)), -1)
    for lecture_no in range(lecture_total_number):
        for reviewees_no in range(len(reviewees)):
            this_reviewer = assign_history[lecture_no][reviewees_no]
            table_converted[lecture_no][reviewers.index(this_reviewer)] = reviewees[
                reviewees_no
            ]
    return table_converted


def convert_assigntable_reviewer_to_reviewee(reviewees, reviewers, assign_history):
    """
    Convert assign table from the reviewer's view to the reviewee's view.

    Args:
        reviewees (List, len=M): The list of reviewees number.
        reviewers (List, len=N): The list of reviewers number.
        assign_history (numpy.array, shape=(EX, N)): Reviewer assignment's history.

    Returns:
        table_converted (numpy.array, shape=(EX, M)): Converted reviewer assignment's history.

    Notes:
        The length of list reviewers and assign_history.shape[1] must be equal.

    Examples:
        >>> reviewees = [1, 2, 3]
        >>> reviewers = [11, 12, 13, 14, 15]
        >>> assign_history = np.array([
                [1, 2, 3, -1, -1],
                [3, -1, -1, 1, 2],
                [-1, 1, 2, 3, -1],
                [2, 3, -1, -1, 1]
            ])
        >>> convert_assigntable_reviewer_to_reviewee(reviewees, reviewers, assign_history)
            # np.array([
            #   [11, 12, 13],
            #   [14, 15, 11],
            #   [12, 13, 14],
            #   [15, 11, 12]
            # ])
    """
    assert (
        len(reviewers) == assign_history.shape[1]
    ), "Mismatch the number of reviewers."
    lecture_total_number = assign_history.shape[0]
    table_converted = np.full((lecture_total_number, len(reviewees)), -1)
    for lecture_no in range(lecture_total_number):
        for reviewers_no in range(len(reviewers)):
            this_reviewee = assign_history[lecture_no][reviewers_no]
            if this_reviewee == -1:
                continue
            table_converted[lecture_no][reviewees.index(this_reviewee)] = reviewers[
                reviewers_no
            ]
    return table_converted


def count_duplicates_the_same_index(target_array, org_array):
    """
    Count the number of columns with duplicate values.

    Args:
        target_array (numpy.array): Target array.
        org_array (numpy.array): Original array.

    Returns:
        cnt (int): The number of columns with duplicate value.

    Notes:
        target_array.shape[1] and org_array.shape[1] must be equal.

    Examples:
        >>> target_array = np.array([13, 15, 14])
            # arget_array = np.array([[13, 15, 14]]) is ok.
        >>> org_array = np.array([
                [11, 12, 13],
                [14, 15, 11],
                [12, 13, 14],
                [15, 11, 12]
            ])
        >>> count_duplicates_the_same_index(target_array, org_array)
            # 2
    """
    if target_array.ndim == 1:
        target_array = target_array.reshape(1, -1)
    concat_arr = np.concatenate([target_array, org_array])
    # concat_arr = np.vstack(target_array, org_array)
    cnt = 0
    for col in range(concat_arr.shape[1]):
        tmp_arr = concat_arr[:, col]
        if np.unique(tmp_arr).size != tmp_arr.size:
            cnt += 1
    return cnt


def assign_random(reviewees, reviewers, assign_history=None, max_iter=10000):
    """
    Assign reviewers by roll.

    Args:
        reviewees (List, len=M): The list of reviewees number.
        reviewers (List, len=N): The list of reviewers number.
        assign_history (numpy.array, shape=(EX, M), default: None):
            Reviewer assignment's history.
        max_iter (int, default: 10000): Maximum number of iterations of random draws.

    Returns:
        new_assign (numpy.array, shape=(1, M)): New assignment of reviewers.

    Note:
        Reviewers must be more than reviewees; len(reviewees) <= len(reviewers).
    """
    assert len(reviewees) <= len(reviewers), "Reviewers must be more than reviewees."

    if assign_history is None:
        random.shuffle(reviewers)
        best_new_assign = np.array(reviewers[: len(reviewees)])
        return best_new_assign

    history_list_flatten = list(assign_history.flatten()) + reviewers
    min_cnt_duplicates = len(reviewees)
    best_new_assign = []
    for i in range(max_iter):
        random.shuffle(history_list_flatten)

        # select reviewer
        c = collections.Counter(history_list_flatten)
        values, counts = zip(*c.most_common()[::-1])
        new_array = np.array(values[: len(reviewees)])
        cnt_duplicates = count_duplicates_the_same_index(new_array, assign_history)
        if cnt_duplicates == 0:
            return new_array
        elif i == 0 or cnt_duplicates < min_cnt_duplicates:
            best_new_assign = new_array
            min_cnt_duplicates = cnt_duplicates
    return best_new_assign


def assign_roll_students(students, assign_history=None):
    """
    Assign reviewers by roll.

    Args:
        reviewees (List, len=M): The list of reviewees number.
        assign_history (numpy.array, shape=(EX, M), default: None):
            Reviewer assignment's history.

    Returns:
        new_assign (numpy.array, shape=(1, M)): New assignment of reviewers.
    """
    if assign_history is None:
        new_assign = np.roll(np.array(students), 1)
        return new_assign

    lecture_total_number = assign_history.shape[0]
    shift = (lecture_total_number % (len(students) - 1)) + 1
    new_assign = np.roll(np.array(students), shift)
    return new_assign


def assign(users):
    """
    Assign next reviewers.

    Args:
        users (pandas.core.frame.DataFrame): Dataframe of users.

    Returns:
        users (pandas.core.frame.DataFrame): Dataframe of users with new assigned reviewers.
    """
    groups = list(set(users["group"]))
    all_github_accounts = list(users["github_account"])
    students = get_list_from_df(users, "group", "student", "github_account")
    students_numbers = list(range(len(students)))

    columns_list = users.columns.values
    lecture_number = len(columns_list) - 3
    first_assign = False
    if lecture_number == 1:
        first_assign = True
    for group in groups:
        users_in_group = users.query(f"group == '{group}'")
        reviewers = get_list_from_df(users_in_group, "group", group, "github_account")
        reviewers_numbers = list(range(len(reviewers)))

        reviewers_numbers_copy = copy.deepcopy(reviewers_numbers)
        if first_assign:
            assign_history = None
        else:
            assign_history = [
                convert_list_with_correspondence_lists(
                    list(users_in_group[col]), students, students_numbers
                )
                for col in columns_list[4:]
            ]
            assign_history = convert_assigntable_reviewer_to_reviewee(
                students_numbers, reviewers_numbers, np.array(assign_history)
            )
        if group == "student":
            new_assign = assign_roll_students(students_numbers, assign_history)
        else:
            new_assign = assign_random(
                students_numbers, reviewers_numbers_copy, assign_history
            )
        new_assign = new_assign.reshape((1, -1))
        if first_assign:
            assign_history_with_new = new_assign
        else:
            assign_history_with_new = np.concatenate([assign_history, new_assign])
        assign_history_with_new = convert_assigntable_reviewee_to_reviewer(
            students_numbers, reviewers_numbers, assign_history_with_new
        )
        new_assign = convert_list_with_correspondence_lists(
            list(assign_history_with_new[-1]),
            students_numbers,
            students,
        )
        for ass, reviewer in zip(new_assign, reviewers):
            users.at[all_github_accounts.index(reviewer), f"EX{lecture_number}"] = ass
    return users


def main():
    """Run main function for assignment."""
    args = parse_args()
    users = pd.read_csv(args.users)
    users = assign(users)
    users.to_csv(args.users, index=False)


if __name__ == "__main__":
    main()
