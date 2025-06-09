	.file	"4_race_cond.c"
	.text
	.globl	pings
	.bss
	.align 4
	.type	pings, @object
	.size	pings, 4
pings:
	.zero	4
	.text
	.globl	_Z7routinePv
	.type	_Z7routinePv, @function
_Z7routinePv:
.LFB26:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -24(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L2
.L3:
	movl	pings(%rip), %eax     # load pings val from memory to cpu register - %rip is instruction pointer
	addl	$1, %eax              # add 1 to value in eax 
	movl	%eax, pings(%rip)     # store incremented valuee  - when threads are running parallel, this is a place where race condition happens 
	addl	$1, -4(%rbp)          # increment loop counter
.L2:
	cmpl	$9999, -4(%rbp)
	jle	.L3
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE26:
	.size	_Z7routinePv, .-_Z7routinePv
	.section	.rodata
.LC0:
	.string	"Number of pings : %d\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB27:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	leaq	-24(%rbp), %rax
	movl	$0, %ecx
	leaq	_Z7routinePv(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L6
	movl	$1, %eax
	jmp	.L11
.L6:
	leaq	-16(%rbp), %rax
	movl	$0, %ecx
	leaq	_Z7routinePv(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L8
	movl	$2, %eax
	jmp	.L11
.L8:
	leaq	-24(%rbp), %rax
	movl	$0, %ecx
	leaq	_Z7routinePv(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L9
	movl	$3, %eax
	jmp	.L11
.L9:
	leaq	-24(%rbp), %rax
	movl	$0, %ecx
	leaq	_Z7routinePv(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L10
	movl	$4, %eax
	jmp	.L11
.L10:
	movl	pings(%rip), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %eax
.L11:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L12
	call	__stack_chk_fail@PLT
.L12:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE27:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
