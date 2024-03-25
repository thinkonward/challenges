from typing import List, Union
import json
import pandas as pd
import numpy as np
from sympy import Interval, Union, Intersection
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

###Data process
class DataObject:
    """
    A class representing data objects with name, data, and optional intervals (actual and predicted).
    """

    def __init__(self, name: str, data: pd.DataFrame) -> None:
        """
        Initializes a DataObject instance.

        Args:
            name: The name of the data object (character).
            data: A pandas DataFrame representing the data.
        """

        self.name = name
        self.data = data
        self.actual_intervals: List[Union[int, float]] = []
        self.predicted_intervals: List[Union[int, float]] = []

    def assign_intervals(self, intervals: List[tuple], is_actual: bool = True) -> None:
        """
        Assigns a list of intervals (start, stop) to the object.

        Args:
            intervals: A list of tuples representing intervals (start, stop).
            is_actual: A boolean flag indicating whether the intervals are actual (True) or predicted (False).
        """

        if is_actual:
            self.actual_intervals = intervals
        else:
            self.predicted_intervals = intervals

def add_intervals(new_data_files: List[str], new_intervals: List[List[tuple]], is_actual: bool = False, current_solutions: List[DataObject] = []) -> List[DataObject]:
    """
    Adds new DataObjects to the solutions list based on provided data files and intervals.

    Args:
        new_data_files: A list of filenames for the data to include in the new objects.
        new_intervals: A list of lists of intervals (start, stop) for the new objects.
        is_actual: A boolean flag indicating whether the intervals are actual (True) or predicted (False).
        current_solutions: A list of existing DataObject instances (default: empty list).

    Returns:
        A list of DataObjects, including the current solutions and the newly created ones.
    """

    solutions = current_solutions.copy()
    for name, data_file, intervals in zip(new_data_files, new_data_files, new_intervals):
        # Assuming data can be loaded from filenames (adjust based on your data source)
        data = pd.read_csv(data_file,sep="\t")
        data = data[['time','y']]
        data_object = DataObject(name, data)
        data_object.assign_intervals(intervals, is_actual=is_actual)
        solutions.append(data_object)

    return solutions

import os
def get_actual_list_intervals_files(data_object_dict: dict[str, List[tuple]]) -> List[str]:
    """
    Extracts list of file names for predicted intervals from a dictionary of DataObjects.

    Args:
      data_object_dict: A dictionary with names (with .csv extension) as keys and predicted intervals as values (output of dict2DataObjectList).

    Returns:
      A list of file names (without .csv extension) for predicted interval files.
    """

    actual_list_intervals_files = []
    for file_name, _ in data_object_dict.items():
      # Split the file name with extension
      name_without_extension = os.path.splitext(file_name)[0]
      # Add '_intervals' suffix and append to the list
      actual_list_intervals_files.append(name_without_extension + "_intervals.csv")

    return actual_list_intervals_files

def create_submission_dictionary(data_object_solutions: List[DataObject]) -> dict[str, List[tuple]]:
    """
    Creates a dictionary from DataObject objects, extracting name-predicted_intervals pairs.

    Args:
        data_object_solutions: A list of DataObject instances, each instance including attribute name and predicted_intervals.

    Returns:
        A dictionary with keys as names and values as corresponding predicted intervals.
    """

    submission_dict = {}
    for object in data_object_solutions:
        submission_dict[object.name] = object.predicted_intervals

    return submission_dict

def save_submission_to_json(submission_dict: dict, filename: str) -> None:
    """
    Saves a submission dictionary to a JSON file.

    Args:
        submission_dict: A dictionary with names as keys and predicted intervals as values.
        filename: The desired filename for the JSON file.
    """

    with open(filename, "w") as f:
        json.dump(submission_dict, f, indent=4)
        
def sample_submission_generator(data_object_solutions: List[DataObject], filename: str) ->None:
    """
    Saves a submission JSON file from a solution, list of DataObject instances.

    Args:
        data_object_solutions: A list of DataObject instances, each instance including attribute name and predicted_intervals.
        filename: The desired filename for the JSON file.
    """
    submission_dict = create_submission_dictionary(data_object_solutions)
    save_submission_to_json(submission_dict,filename)

def register_actual_intervals(solutions: List[DataObject], actual_intervals: List[List[tuple]]) -> List[DataObject]:
    """
    Adds actual intervals to each DataObject in the solutions list.

    Args:
        solutions: A list of DataObject instances.
        actual_intervals: A list of lists of actual intervals (start, stop) corresponding to each DataObject.

    Returns:
        A list of DataObject instances with updated actual_intervals attributes.
    """

    if len(solutions) != len(actual_intervals):
        raise ValueError("Number of solutions and actual intervals must be equal.")

    for solution, intervals in zip(solutions, actual_intervals):
        solution.actual_intervals = intervals

    return solutions


###Model example
def get_local_extrema_with_distance(data: pd.DataFrame, distance: int = 1) -> pd.DataFrame:
  """
  Finds local minima and maxima in a DataFrame with columns "time" and "y" using find_peaks with minimum distance.

  Args:
      data: A pandas DataFrame with columns "time" and "y" representing the data series.
      distance: Minimum distance (number of samples) between peaks (defaults to 1).

  Returns:
      A pandas DataFrame with columns "time", "value", and "type" containing the coordinates and type ('min' or 'max') of local extrema.
  """

  peaks, _ = find_peaks(data["y"].values, height=None, distance=distance)  # Consider height=None for all peaks

  # Identify minima by inverting y values and finding maxima
  anti_peaks, _ = find_peaks(-data["y"].values, height=None, distance=distance)
  minima_indices = data.index[anti_peaks]

  all_extrema = pd.DataFrame(columns=["time", "value", "type"])

  # Add maxima data
  maxima_df = pd.DataFrame({"time": data["time"].iloc[peaks], "value": data["y"].iloc[peaks], "type": "max"})

  # Add minima data (inverted y values for clarity)
  minima_df = pd.DataFrame({"time": data["time"].iloc[minima_indices], "value": -data["y"].iloc[minima_indices], "type": "min"})

  # Combine maxima and minima DataFrames
  all_extrema = pd.concat([all_extrema, maxima_df, minima_df], ignore_index=True)

  return all_extrema

class SimpleModel:
    """
    A class representing models as created here, adding the predict function.
    """

    def __init__(self, avg_interval_length: float, avg_time_diff: float) -> None:
        """
        Initializes a SimpleModel instance.

        Args:
            avg_interval_length (float).
            avg_time_diff (float).
        """

        self.avg_interval_length = avg_interval_length
        self.avg_time_diff = avg_time_diff

    def fit(self, data_object_list: List[DataObject]) -> None:
        """
        Train the model: calculates average interval length and time difference between intervals across DataObjects.

        Args:
        data_object_list: A list of DataObject instances with actual_intervals attributes.

        Returns:
          A tuple containing two elements:
          - Average interval length (average stop-start across all actual_intervals).
          - Average time difference between intervals (average start_next - stop_current).
        """
        all_interval_lengths = []
        all_time_diffs = []

        for obj in data_object_list:
          actual_intervals = obj.actual_intervals
          for i in range(len(actual_intervals) - 1):
            # Assuming actual_intervals contain tuples (start, stop)
            current_interval_length = actual_intervals[i][1] - actual_intervals[i][0]
            next_interval_start = actual_intervals[i + 1][0]
            time_diff = next_interval_start - actual_intervals[i][1]
            all_interval_lengths.append(current_interval_length)
            all_time_diffs.append(time_diff)
        avg_interval_length = np.mean(all_interval_lengths) if all_interval_lengths else 0
        avg_time_diff = np.mean(all_time_diffs) if all_time_diffs else 0
        self.avg_interval_length = avg_interval_length
        self.avg_time_diff = avg_time_diff
        
    def predict(self, instance: DataObject):
        """
        Predict a list of intervals (start, stop) to the instance.

        Args:
            instance: A DataObject.
        """
        
        model_distance = self.avg_interval_length + self.avg_time_diff
        local_extrema=get_local_extrema_with_distance(instance.data, distance=model_distance)
        def get_predicted_intervals(local_extrema: pd.DataFrame) -> List[tuple]:
          """
          Extracts predicted intervals (start, stop) from a DataFrame containing local extrema.

          Args:
              local_extrema: A pandas DataFrame with columns "time", "value", and "type" (output of get_local_extrema_with_distance).

          Returns:
              A list of tuples representing predicted intervals (start, stop).
          """
          local_extrema.sort_values(by=['time'],inplace=True)
          predicted_intervals = []
          current_type = None
          start = None

          for index, row in local_extrema.iterrows():
            if current_type is None:  # First iteration
              current_type = row["type"]
              start = row["time"]
            elif row["type"] == current_type:  # Continue same interval
              start = row["time"]
            else:  # Type changed, create interval
              stop = row["time"]
              predicted_intervals.append((start, stop))
              current_type = row["type"]
              start = row["time"]  # Prepare for next interval

          # Handle potential last interval
          if start is not None:
            predicted_intervals.append((start, local_extrema["time"].iloc[-1]))

          return predicted_intervals
        
        return get_predicted_intervals(local_extrema)
    
def predict(model, data_object_list: List[DataObject]) -> List[DataObject]:
  """
  Predicts intervals for each DataObject instance using the provided model.

  Args:
      model: A machine learning model capable of predicting intervals.
      data_object_list: A list of DataObject instances with data attribute.

  Returns:
      A list of DataObject instances with predicted intervals filled using the model.
  """

  predicted_data_object_list = []
  for obj in data_object_list:
    # Assuming the model predicts a list of tuples representing intervals (start, stop)
    predicted_intervals = model.predict(obj)
    new_object = DataObject(obj.name, obj.data)  # Create a copy to avoid modifying original objects
    new_object.predicted_intervals = predicted_intervals
    predicted_data_object_list.append(new_object)

  return predicted_data_object_list


#Metrics:
def calculate_iou_intersection_union(interval1, interval2,output_option="iou"):
        intersection = list(Intersection(Interval(interval1[0],interval1[1]),Interval(interval2[0],interval2[1])).boundary)
        if len(intersection)>1:
            intersection = abs(intersection[1]-intersection[0])
        else:
            intersection = 0
        union = list(Union(Interval(interval1[0],interval1[1]),Interval(interval2[0],interval2[1])).boundary)
        if len(union)==2:
            union = abs(union[1]-union[0])
        elif len(union)==4:
            union = abs(union[1]-union[0]+union[3]-union[2])
        else:
            union = 0
        if output_option == "iou":
            return float(intersection / union) if (union>0) else 0
        elif output_option == "intersection":
            return float(intersection)
        elif output_option == "union":
            return float(union)
        else:
            raise ValueError("Invalid output_option. Choose from 'iou', 'intersection', or 'union'.")
            
def assign_intervals_and_calculate_iou(true_intervals, predicted_intervals, output_option="average_iou"):
    """
    Assigns predicted intervals to true intervals based on highest IoU and calculates relevant metrics.

    Args:
        true_intervals: A list of tuples representing true intervals (start, end).
        predicted_intervals: A list of tuples representing predicted intervals (start, end).
        output_option: String specifying the desired output: "average_iou", "assigned_pairs", "unassigned_predicted", or "unassigned_real".

    Returns:
        The corresponding output based on the selected output_option.
        average_iou computed as: (sum of intersection/ sum of unions) of real-pred pairs
        option assigned_pairs gives output results as each assignation as a list with tuples (true_start, true_end, best_predicted_interval, best_iou, diff_starts, exp_diff_starts) 
        with exp_diff_starts= exp(-diff_starts/tau) with diff_starts=abs(true_start-predicted_start), tau=5 if predicted_start > true_start and tau=10 otherwise.
    """

    assigned_pairs = []
    unassigned_predicted = predicted_intervals.copy()
    total_intersection = 0
    total_union = 0
    unassigned_real = []

    for true_start, true_end in true_intervals:
        best_iou = 0
        best_intersection = 0
        best_predicted_interval = None

        for i, (predicted_start, predicted_end) in enumerate(unassigned_predicted):
            current_inters = calculate_iou_intersection_union((true_start, true_end), (predicted_start, predicted_end),'intersection')
            current_iou = calculate_iou_intersection_union((true_start, true_end), (predicted_start, predicted_end),'iou')
            if current_iou > best_iou:
                best_intersection = current_inters
                best_iou = current_iou
                best_predicted_interval = (predicted_start, predicted_end)
                diff_starts = abs(true_start-predicted_start)
                tau=10
                if predicted_start>true_start: tau = 5
                exp_diff_starts = np.exp(-diff_starts/tau)

        if best_predicted_interval:
            assigned_pairs.append((true_start, true_end, best_predicted_interval,best_iou,diff_starts,exp_diff_starts))
            total_intersection += best_intersection
            best_union = calculate_iou_intersection_union((true_start, true_end), best_predicted_interval,'union')
            total_union += best_union
            unassigned_predicted.remove(best_predicted_interval)
        else:
            unassigned_real.append((true_start,true_end))

    if output_option == "average_iou":
        return total_intersection/total_union if ((total_union>0) & (total_intersection>0)) else 0
    elif output_option == "assigned_pairs":
        return assigned_pairs
    elif output_option == "unassigned_predicted":
        return unassigned_predicted
    elif output_option == "unassigned_real":
        return unassigned_real
    else:
        raise ValueError("Invalid output_option. Choose from 'average_iou', 'assigned_pairs', 'unassigned_predicted', or 'unassigned_real'.")
        
def get_metrics(true_intervals, predicted_intervals, output_option="final_score"):
    """
    Compute metrics for predicted intervals matching true intervals.
    
    Args:
        true_intervals: A list of tuples representing true intervals (start, end).
        predicted_intervals: A list of tuples representing predicted intervals (start, end).
        output_option: String specifying the desired output: "final_score", "average_iou", "NegExpDiffStarts", "Recall", or "Precision".

    Returns:
        The corresponding output based on the selected output_option.
        average_iou computed as: (sum of intersection/ sum of unions) of real-pred pairs
        option NegExpDiffStarts returns the average across matches of exp(-diff_starts/5) 
        with diff_starts=0 if abs(true_start-predicted_start)<=5 and abs(true_start-predicted_start) else
    """
    assignations= assign_intervals_and_calculate_iou(true_intervals, predicted_intervals, output_option="assigned_pairs")
    if output_option == "final_score":
        IoU = assign_intervals_and_calculate_iou(true_intervals, predicted_intervals, output_option="average_iou")
        NegExpDiffStarts = 0
        if(len(assignations)>0):
            NegExpDiffStarts = np.mean(list(map(lambda i: assignations[i][5], range(0,len(assignations)))))
        Recall=float(len(assignations)/len(true_intervals)) if (len(true_intervals)>0) else 0
        Precision=float(len(assignations)/len(predicted_intervals)) if (len(predicted_intervals)>0) else 0
        FBeta=float((1+0.5*0.5)*(Precision*Recall)/(0.5*0.5*Precision+Recall)) if((Recall>0) | (Recall>0)) else 0
        final_score = (IoU + 2*NegExpDiffStarts + 3*FBeta)/6
        return final_score
    elif output_option == "average_iou":
        IoU = assign_intervals_and_calculate_iou(true_intervals, predicted_intervals, output_option="average_iou")
        return IoU
    elif output_option == "NegExpDiffStarts":
        NegExpDiffStarts = np.mean(list(map(lambda i: assignations[i][5], range(0,len(assignations)))))
        return NegExpDiffStarts
    elif output_option == "Recall":
        Recall=float(len(assignations)/len(true_intervals)) if (len(true_intervals)>0) else 0
        return Recall
    elif output_option == "Precision":
        Precision=float(len(assignations)/len(predicted_intervals)) if (len(predicted_intervals)>0) else 0
        return Precision
    else:
        raise ValueError("Invalid output_option. Choose from 'final_score', 'average_iou', 'NegExpDiffStarts', 'Recall', or 'Precision' .")
        
def get_avg_metrics(data_object_list: List[DataObject], metric_option="final_score") -> float:
    """
    Calculates the average of a specified metric across DataObject instances using get_metrics.

    Args:
        data_object_list: A list of DataObject instances with actual_intervals and predicted_intervals attributes.
        metric_option: String specifying the desired metric to compute, valid options are:
                       final_score, average_iou, NegExpDiffStarts, Recall, or Precision.

    Returns:
        The average of the specified metric across all DataObject instances.
    """

    all_scores = []
    for obj in data_object_list:
        score = get_metrics(obj.actual_intervals, obj.predicted_intervals, output_option=metric_option)
        all_scores.append(score)

    avg_score = np.mean(all_scores)
    return avg_score
        
#Visualization of a weil
def plot_well(intervals,data,start=0,end=1576800,title="Results",s=0.01):
  """
  Plot time series data with y axis 'y' from df and color by label according to intervals.
  Allow zooming to [initio,end] times

  Args:
      intervals: A list of tuples representing intervals, where each tuple is (start, stop).
      data: DataFrame including 'time' and 'y' values.
      initio, end: the time interval in which the plot zooms.
      title: text you can specify for the title
      s: size of the plot points.

  Returns:
      A time series plot.
  """  
  final_time=len(data)
    
  def intervals2dataframe(intervals, final_time):
      """
      Converts a list of intervals to a DataFrame with time and label columns.

      Args:
      intervals: A list of tuples representing intervals, where each tuple is (start, stop).
      final_time: The final time point for the DataFrame.

      Returns:
      A DataFrame with columns "time" and "label".
      """

      # Create an empty DataFrame
      df = pd.DataFrame({'time': range(final_time)})

      # Set the default label to 0
      df['label'] = 0

      # Loop through each interval and set the label to 1 within the interval
      for start, stop in intervals:
        df.loc[start:stop - 1, 'label'] = 1

      return df

  df = intervals2dataframe(intervals, final_time)

  plt.scatter(df[start:end].time,data[start:end].y,color=np.where(df[start:end].label, 'r','C0'),s=s)
  plt.xlabel("Time")
  plt.ylabel("Time Series")
  plt.title(title)
  plt.show()
