package inside_payment.service;

import edu.fudan.common.util.Response;
import inside_payment.entity.*;
import org.springframework.http.HttpHeaders;


/**
 * @author Administrator
 * @date 2017/6/20.
 */
public interface InsidePaymentService {

    /**
     * pay by payment info
     *
     * @param info payment info
     * @param headers headers
     * @return Response
     */
    Response pay(PaymentInfo info , HttpHeaders headers);

    /**
     * create account by payment info
     *
     * @param info payment info
     * @param headers headers
     * @return Response
     */
    Response createAccount(AccountInfo info, HttpHeaders headers);

    /**
     * add money with user id, money
     *
     * @param userId user id
     * @param  money money
     * @param headers headers
     * @return Response
     */
    Response addMoney(String userId,String money, HttpHeaders headers);

    /**
     * query payment info
     *
     * @param headers headers
     * @return Response
     */
    Response queryPayment(HttpHeaders headers);

    /**
     * query account info
     *
     * @param headers headers
     * @return Response
     */
    Response queryAccount(HttpHeaders headers);

    /**
     * drawback with user id, money
     *
     * @param userId user id
     * @param  money money
     * @param headers headers
     * @return Response
     */
    Response drawBack(String userId, String money, HttpHeaders headers);

    /**
     * pay difference by payment info
     *
     * @param info payment info
     * @param headers headers
     * @return Response
     */
    Response payDifference(PaymentInfo info, HttpHeaders headers);

    /**
     * query add money
     *
     * @param headers headers
     * @return Response
     */
    Response queryAddMoney(HttpHeaders headers);

    /**
     * init payment
     *
     * @param payment payment
     * @param headers headers
     * @return Response
     */
    void initPayment(Payment payment, HttpHeaders headers);

}
