package inside_payment.repository;

import inside_payment.entity.Payment;
import org.springframework.data.repository.CrudRepository;

import java.util.List;

/**
 * @author fdse
 */
public interface PaymentRepository extends CrudRepository<Payment,String> {

    /**
     * find by id
     *
     * @param id id
     * @return Payment
     */
    Payment findById(String id);

    /**
     * find by order id
     *
     * @param orderId order id
     * @return List<Payment>
     */
    List<Payment> findByOrderId(String orderId);

    /**
     * find all
     *
     * @return List<Payment>
     */
    @Override
    List<Payment> findAll();

    /**
     * find by user id
     *
     * @param userId user id
     * @return List<Payment>
     */
    List<Payment> findByUserId(String userId);
}
