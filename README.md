# Hand-gesture-controlled-cursor
      -Project hosted by electronics club at IITG.
Hand gesture controlled mouse cursor with webcam.

This approach for hand gesture recognition is vision based,which
uses image processing techniques and inputs from a computer webcam.
The input frame captured from webcam is processwed in three part-

--skin detection(using histogram model)

--hand gesture detection(using contour extraction,convex hull detection) 

--Integrating mouse function(using Pyautogui library)

