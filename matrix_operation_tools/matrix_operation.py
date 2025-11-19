import numpy as np
import os

class MatrixOperationsTool:
    def __init__(self):
        self.matrices = {}
        
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_header(self):
        """Display application header"""
        print("=" * 60)
        print("           MATRIX OPERATIONS TOOL")
        print("=" * 60)
        print("Built with Python & NumPy")
        print("=" * 60)
    
    def input_matrix(self, name):
        """Get matrix input from user"""
        print(f"\nINPUT MATRIX {name}")
        print("-" * 30)
        
        try:
            rows = int(input("Enter number of rows: "))
            cols = int(input("Enter number of columns: "))
            
            if rows <= 0 or cols <= 0:
                print("Error: Rows and columns must be positive integers!")
                return None
            
            print(f"\nEnter matrix elements row-wise (space-separated):")
            matrix = []
            
            for i in range(rows):
                while True:
                    row_input = input(f"Row {i+1}: ").strip()
                    elements = row_input.split()
                    
                    if len(elements) != cols:
                        print(f"Error: Please enter exactly {cols} elements!")
                        continue
                    
                    try:
                        row = [float(x) for x in elements]
                        matrix.append(row)
                        break
                    except ValueError:
                        print("Error: Please enter valid numbers!")
            
            np_matrix = np.array(matrix)
            self.matrices[name] = np_matrix
            print(f"Matrix {name} stored successfully!")
            return np_matrix
            
        except ValueError:
            print("Error: Please enter valid integers for rows and columns!")
            return None
    
    def display_matrix(self, matrix, name="Matrix"):
        """Display matrix in a formatted way"""
        print(f"\n{name}:")
        print("-" * 40)
        if matrix is None:
            print("No matrix to display")
            return
        
        rows, cols = matrix.shape
        for i in range(rows):
            row_str = "| "
            for j in range(cols):
                if matrix[i, j].is_integer():
                    row_str += f"{int(matrix[i, j]):>8}"
                else:
                    row_str += f"{matrix[i, j]:>8.2f}"
            row_str += " |"
            print(row_str)
        print("-" * 40)
        print(f"Shape: {rows} x {cols}")
    
    def display_all_matrices(self):
        """Display all stored matrices"""
        if not self.matrices:
            print("\nNo matrices stored yet!")
            return
        
        print("\nSTORED MATRICES:")
        print("=" * 50)
        for name, matrix in self.matrices.items():
            self.display_matrix(matrix, f"Matrix '{name}'")
    
    def matrix_addition(self):
        """Perform matrix addition"""
        print("\nMATRIX ADDITION")
        print("-" * 30)
        
        if len(self.matrices) < 2:
            print("Need at least 2 matrices for addition!")
            return
        
        self.display_all_matrices()
        
        try:
            name1 = input("\nEnter name of first matrix: ").strip()
            name2 = input("Enter name of second matrix: ").strip()
            
            if name1 not in self.matrices or name2 not in self.matrices:
                print("Error: One or both matrices not found!")
                return
            
            A = self.matrices[name1]
            B = self.matrices[name2]
            
            if A.shape != B.shape:
                print("Error: Matrices must have the same dimensions for addition!")
                return
            
            result = A + B
            
            print(f"\nADDITION RESULT: {name1} + {name2}")
            self.display_matrix(A, f"Matrix {name1}")
            self.display_matrix(B, f"Matrix {name2}")
            self.display_matrix(result, "RESULT")
            
            result_name = f"({name1}_plus_{name2})"
            self.matrices[result_name] = result
            print(f"Result stored as '{result_name}'")
            
        except Exception as e:
            print(f"Error during addition: {e}")
    
    def matrix_subtraction(self):
        """Perform matrix subtraction"""
        print("\nMATRIX SUBTRACTION")
        print("-" * 30)
        
        if len(self.matrices) < 2:
            print("Need at least 2 matrices for subtraction!")
            return
        
        self.display_all_matrices()
        
        try:
            name1 = input("\nEnter name of first matrix: ").strip()
            name2 = input("Enter name of second matrix: ").strip()
            
            if name1 not in self.matrices or name2 not in self.matrices:
                print("Error: One or both matrices not found!")
                return
            
            A = self.matrices[name1]
            B = self.matrices[name2]
            
            if A.shape != B.shape:
                print("Error: Matrices must have the same dimensions for subtraction!")
                return
            
            result = A - B
            
            print(f"\nSUBTRACTION RESULT: {name1} - {name2}")
            self.display_matrix(A, f"Matrix {name1}")
            self.display_matrix(B, f"Matrix {name2}")
            self.display_matrix(result, "RESULT")
            
            result_name = f"({name1}_minus_{name2})"
            self.matrices[result_name] = result
            print(f"Result stored as '{result_name}'")
            
        except Exception as e:
            print(f"Error during subtraction: {e}")
    
    def matrix_multiplication(self):
        """Perform matrix multiplication"""
        print("\nMATRIX MULTIPLICATION")
        print("-" * 30)
        
        if len(self.matrices) < 2:
            print("Need at least 2 matrices for multiplication!")
            return
        
        self.display_all_matrices()
        
        try:
            name1 = input("\nEnter name of first matrix: ").strip()
            name2 = input("Enter name of second matrix: ").strip()
            
            if name1 not in self.matrices or name2 not in self.matrices:
                print("Error: One or both matrices not found!")
                return
            
            A = self.matrices[name1]
            B = self.matrices[name2]
            
            if A.shape[1] != B.shape[0]:
                print(f"Error: Columns of first matrix ({A.shape[1]}) must equal rows of second matrix ({B.shape[0]})!")
                return
            
            result = np.dot(A, B)
            
            print(f"\nMULTIPLICATION RESULT: {name1} x {name2}")
            self.display_matrix(A, f"Matrix {name1}")
            self.display_matrix(B, f"Matrix {name2}")
            self.display_matrix(result, "RESULT")
            
            result_name = f"({name1}_times_{name2})"
            self.matrices[result_name] = result
            print(f"Result stored as '{result_name}'")
            
        except Exception as e:
            print(f"Error during multiplication: {e}")
    
    def matrix_transpose(self):
        """Perform matrix transpose"""
        print("\nMATRIX TRANSPOSE")
        print("-" * 30)
        
        if not self.matrices:
            print("No matrices available for transpose!")
            return
        
        self.display_all_matrices()
        
        try:
            name = input("\nEnter name of matrix to transpose: ").strip()
            
            if name not in self.matrices:
                print("Error: Matrix not found!")
                return
            
            A = self.matrices[name]
            result = A.T
            
            print(f"\nTRANSPOSE RESULT: {name}^T")
            self.display_matrix(A, "Original Matrix")
            self.display_matrix(result, "Transposed Matrix")
            
            result_name = f"transpose_of_{name}"
            self.matrices[result_name] = result
            print(f"Result stored as '{result_name}'")
            
        except Exception as e:
            print(f"Error during transpose: {e}")
    
    def matrix_determinant(self):
        """Calculate matrix determinant"""
        print("\nMATRIX DETERMINANT")
        print("-" * 30)
        
        if not self.matrices:
            print("No matrices available for determinant calculation!")
            return
        
        self.display_all_matrices()
        
        try:
            name = input("\nEnter name of matrix: ").strip()
            
            if name not in self.matrices:
                print("Error: Matrix not found!")
                return
            
            A = self.matrices[name]
            
            if A.shape[0] != A.shape[1]:
                print("Error: Matrix must be square (n x n) for determinant calculation!")
                return
            
            det = np.linalg.det(A)
            
            print(f"\nDETERMINANT RESULT")
            self.display_matrix(A, f"Matrix {name}")
            print(f"\nDeterminant of {name} = {det:.4f}")
            
            if det == 0:
                print("Note: The matrix is singular (non-invertible)")
            else:
                print("Note: The matrix is invertible")
                
        except Exception as e:
            print(f"Error calculating determinant: {e}")
    
    def matrix_inverse(self):
        """Calculate matrix inverse"""
        print("\nMATRIX INVERSE")
        print("-" * 30)
        
        if not self.matrices:
            print("No matrices available for inverse calculation!")
            return
        
        self.display_all_matrices()
        
        try:
            name = input("\nEnter name of matrix: ").strip()
            
            if name not in self.matrices:
                print("Error: Matrix not found!")
                return
            
            A = self.matrices[name]
            
            if A.shape[0] != A.shape[1]:
                print("Error: Matrix must be square (n x n) for inverse calculation!")
                return
            
            det = np.linalg.det(A)
            if det == 0:
                print("Error: Matrix is singular (determinant = 0), inverse does not exist!")
                return
            
            result = np.linalg.inv(A)
            
            print(f"\nINVERSE RESULT")
            self.display_matrix(A, f"Original Matrix {name}")
            self.display_matrix(result, f"Inverse of {name}")
            
            identity_approx = np.dot(A, result)
            print("\nVerification (A x A^-1 ≈ I):")
            self.display_matrix(identity_approx, "A x A^-1 (Should be Identity)")
            
            result_name = f"inverse_of_{name}"
            self.matrices[result_name] = result
            print(f"Result stored as '{result_name}'")
            
        except Exception as e:
            print(f"Error calculating inverse: {e}")
    
    def delete_matrix(self):
        """Delete a stored matrix"""
        print("\nDELETE MATRIX")
        print("-" * 30)
        
        if not self.matrices:
            print("No matrices to delete!")
            return
        
        self.display_all_matrices()
        
        name = input("\nEnter name of matrix to delete: ").strip()
        
        if name in self.matrices:
            del self.matrices[name]
            print(f"Matrix '{name}' deleted successfully!")
        else:
            print("Error: Matrix not found!")
    
    def show_operations_menu(self):
        """Display main operations menu"""
        print("\nOPERATIONS MENU:")
        print("1.  Matrix Addition")
        print("2.  Matrix Subtraction") 
        print("3.  Matrix Multiplication")
        print("4.  Matrix Transpose")
        print("5.  Matrix Determinant")
        print("6.  Matrix Inverse")
        print("7.  Input New Matrix")
        print("8.  View All Matrices")
        print("9.  Delete Matrix")
        print("0.  Exit")
        print("-" * 30)
    
    def run(self):
        """Main application loop"""
        self.clear_screen()
        self.display_header()
        
        while True:
            self.show_operations_menu()
            
            choice = input("\nEnter your choice (0-9): ").strip()
            
            if choice == '0':
                print("\nThank you for using Matrix Operations Tool!")
                print("Goodbye!")
                break
            
            elif choice == '1':
                self.matrix_addition()
            elif choice == '2':
                self.matrix_subtraction()
            elif choice == '3':
                self.matrix_multiplication()
            elif choice == '4':
                self.matrix_transpose()
            elif choice == '5':
                self.matrix_determinant()
            elif choice == '6':
                self.matrix_inverse()
            elif choice == '7':
                matrix_name = input("Enter a name for this matrix: ").strip()
                self.input_matrix(matrix_name)
            elif choice == '8':
                self.display_all_matrices()
            elif choice == '9':
                self.delete_matrix()
            else:
                print("Invalid choice! Please enter a number between 0-9.")
            
            if choice != '0':
                input("\nPress Enter to continue...")


def demo_mode():
    """Preload some demo matrices for testing"""
    tool = MatrixOperationsTool()
    
    tool.matrices['A'] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tool.matrices['B'] = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    tool.matrices['C'] = np.array([[2, 0], [0, 2]])
    tool.matrices['identity'] = np.eye(3)
    
    print("Demo matrices loaded: A, B, C, identity")
    return tool


if __name__ == "__main__":
    print("Starting Matrix Operations Tool...")
    
    demo_choice = input("Load demo matrices? (y/n): ").strip().lower()
    
    if demo_choice == 'y':
        tool = demo_mode()
        input("Press Enter to continue to main menu...")
    else:
        tool = MatrixOperationsTool()
    
    tool.run()