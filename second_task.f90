! ============================================================
! Module containing the CUDA kernel for matrix transpose
! ============================================================
module transpose_kernels
    use cudafor
    implicit none
contains

    ! Kernel: each thread copies one element from A(i,j) to B(j,i)
    attributes(global) subroutine transpose_matrix_kernel(A, B, m, n)
        real, device, intent(in)  :: A(:,:)
        real, device, intent(out) :: B(:,:)
        integer, value, intent(in) :: m, n   ! dimensions of A (rows=m, cols=n)
        integer :: i, j
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i <= m .and. j <= n) then
            B(j, i) = A(i, j)   ! transpose operation
        end if
    end subroutine transpose_matrix_kernel

end module transpose_kernels

! ============================================================
! Main program: matrix transpose using GPU and CPU verification
! ============================================================
program second_task
    use cudafor
    use transpose_kernels
    implicit none

    ! Problem size
    integer, parameter :: N_ROWS = 1024          ! rows of original matrix
    integer, parameter :: N_COLS = 512           ! columns of original matrix
    integer, parameter :: BLOCK_X = 32           ! threads per block in x direction
    integer, parameter :: BLOCK_Y = 8            ! threads per block in y direction
    integer, parameter :: PREVIEW = 4            ! size of corner to print

    real, allocatable :: A(:,:)                 ! original matrix (host)
    real, allocatable :: B_gpu(:,:)             ! transposed matrix from GPU (host)
    real, allocatable :: B_cpu(:,:)             ! transposed matrix from CPU (host)

    real, device, allocatable :: A_d(:,:)       ! original matrix on device
    real, device, allocatable :: B_d(:,:)       ! transposed matrix on device

    type(dim3) :: grid, block
    integer :: i, j
    real :: tolerance = 1.0e-5

    ! ============================================================
    ! 1. Allocate host memory and fill with random numbers
    ! ============================================================
    allocate(A(N_ROWS, N_COLS))
    allocate(B_gpu(N_COLS, N_ROWS))
    allocate(B_cpu(N_COLS, N_ROWS))

    call random_number(A)           ! uniform random numbers in [0,1)
    A = 100.0 * A                   ! scale to [0,100)

    ! ============================================================
    ! 2. Allocate device memory and copy data
    ! ============================================================
    allocate(A_d(N_ROWS, N_COLS))
    allocate(B_d(N_COLS, N_ROWS))
    A_d = A

    ! ============================================================
    ! 3. Configure kernel launch parameters (2D grid of 2D blocks)
    ! ============================================================
    block = dim3(BLOCK_X, BLOCK_Y, 1)
    grid = dim3(ceiling(real(N_ROWS) / BLOCK_X), &
                ceiling(real(N_COLS) / BLOCK_Y), 1)

    ! ============================================================
    ! 4. Launch the transpose kernel
    ! ============================================================
    call transpose_matrix_kernel<<<grid, block>>>(A_d, B_d, N_ROWS, N_COLS)

    ! Synchronize and check for errors (optional but recommended)
    ! call cudaDeviceSynchronize()

    ! ============================================================
    ! 5. Copy result back to host
    ! ============================================================
    B_gpu = B_d

    ! ============================================================
    ! 6. CPU transpose using explicit loops (verification)
    ! ============================================================
    do i = 1, N_ROWS
        do j = 1, N_COLS
            B_cpu(j, i) = A(i, j)
        end do
    end do

    ! ============================================================
    ! 7. Print a small corner of the matrices for visual inspection
    ! ============================================================
    call print_corner("Original A (first rows, cols)", A, PREVIEW, PREVIEW)
    call print_corner("Transposed B (CPU)", B_cpu, PREVIEW, PREVIEW)
    call print_corner("Transposed B (GPU)", B_gpu, PREVIEW, PREVIEW)

    ! ============================================================
    ! 8. Compare GPU and CPU results
    ! ============================================================
    if (all(abs(B_gpu - B_cpu) <= tolerance)) then
        print *, "SUCCESS: GPU and CPU results match."
    else
        print *, "FAILURE: GPU and CPU results differ."
    end if

    ! ============================================================
    ! 9. Clean up
    ! ============================================================
    deallocate(A, B_gpu, B_cpu)
    deallocate(A_d, B_d)

contains

    ! Helper subroutine to print a corner of a 2D array
    subroutine print_corner(title, x, nrows, ncols)
        character(*), intent(in) :: title
        real,         intent(in) :: x(:,:)
        integer,      intent(in) :: nrows, ncols
        integer :: i, j

        write(*,*)
        write(*,'(a)') trim(title)
        do i = 1, nrows
            write(*,'(*(f10.4,1x))') (x(i, j), j = 1, ncols)
        end do
    end subroutine print_corner

end program second_task
