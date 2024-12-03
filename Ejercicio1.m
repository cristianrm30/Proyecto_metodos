% Método de Descenso por Gradiente

% Definir la matriz A y el vector b
A = [4, 1; 1, 3];
b = [1; 2];

% Inicializar la suposición inicial x0
x = [0; 0];

% Parámetros de tolerancia y máximo de iteraciones
tol = 1e-6;
max_iter = 1000;

% Inicializar residual y contador de iteraciones
r = b - A*x;
iterations = 0;

% Almacenar los resultados
results = zeros(max_iter, 4);

% Iteración del Método de Descenso por Gradiente
while norm(r) > tol && iterations < max_iter
    % Calcular el tamaño del paso alpha
    alpha = (r' * r) / (r' * A * r);
    
    % Actualizar la solución
    x = x + alpha * r;
    
    % Calcular el nuevo residual
    r = b - A*x;
    
    % Guardar los resultados
    iterations = iterations + 1;
    results(iterations, :) = [iterations, x', norm(r)];
end

% Mostrar los resultados en una tabla
T = array2table(results(1:iterations, :), 'VariableNames', {'Iteration', 'x1', 'x2', 'Residual'});
disp(T)

% Graficar la convergencia
figure;
plot(T.Iteration, T.x1, '-o', 'DisplayName', 'x1');
hold on;
plot(T.Iteration, T.x2, '-o', 'DisplayName', 'x2');
xlabel('Iteration');
ylabel('Value');
title('Convergence of Gradient Descent Method');
legend show;
grid on;
