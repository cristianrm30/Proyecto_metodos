% Método de Descenso por Gradiente para un problema no lineal

% Definir la función y su gradiente
f = @(x) x(1)^2 + 3*x(1)*x(2) + 2*x(2)^2;  % Función objetivo
grad_f = @(x) [2*x(1) + 3*x(2); 3*x(1) + 4*x(2)];  % Gradiente

% Inicializar la suposición inicial x0
x = [0; 0];

% Parámetros de tolerancia y máximo de iteraciones
tol = 1e-6;
max_iter = 1000;

% Inicializar el gradiente y contador de iteraciones
grad = grad_f(x);
iterations = 0;

% Almacenar los resultados
results = zeros(max_iter, 4);

% Iteración del Método de Descenso por Gradiente
while norm(grad) > tol && iterations < max_iter
    % Calcular el tamaño del paso alpha
    alpha = 0.01;  % Tamaño de paso fijo, o se puede usar una búsqueda lineal
    
    % Actualizar la solución
    x = x - alpha * grad;
    
    % Calcular el nuevo gradiente
    grad = grad_f(x);
    
    % Guardar los resultados
    iterations = iterations + 1;
    results(iterations, :) = [iterations, x', f(x)];
end

% Mostrar los resultados en una tabla
T = array2table(results(1:iterations, :), 'VariableNames', {'Iteration', 'x1', 'x2', 'Function Value'});
disp(T)

% Graficar la convergencia
figure;
plot(T.Iteration, T.x1, '-o', 'DisplayName', 'x1');
hold on;
plot(T.Iteration, T.x2, '-o', 'DisplayName', 'x2');
xlabel('Iteration');
ylabel('Value');
title('Convergence of Gradient Descent Method for Nonlinear Problem');
legend show;
grid on;
